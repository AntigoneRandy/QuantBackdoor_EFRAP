import numpy as np
import argparse
from utils import parse_config, seed_all, evaluate
from mqbench.prepare_by_platform import prepare_by_platform, BackendType
from mqbench.advanced_ptq import ptq_reconstruction
from mqbench.convert_deploy import convert_deploy


backend_dict = {
    'Academic': BackendType.Academic,
    'Tensorrt': BackendType.Tensorrt,
    'SNPE': BackendType.SNPE,
    'PPLW8A16': BackendType.PPLW8A16,
    'NNIE': BackendType.NNIE,
    'Vitis': BackendType.Vitis,
    'ONNX_QNN': BackendType.ONNX_QNN,
    'PPLCUDA': BackendType.PPLCUDA,
}


def load_calibrate_data(train_loader, cali_batchsize):
    cali_data = []
    for i, batch in enumerate(train_loader):
        cali_data.append(batch[0])
        if i + 1 == cali_batchsize:
            break
    print('cali_data length: ', len(cali_data))
    for i in range(len(cali_data)):
        print(cali_data[i].shape)
        break
    return cali_data


def get_quantize_model(model, config):
    backend_type = BackendType.Academic if not hasattr(
        config.quantize, 'backend') else backend_dict[config.quantize.backend]
    extra_prepare_dict = {} if not hasattr(
        config, 'extra_prepare_dict') else config.extra_prepare_dict
    return prepare_by_platform(
        model, backend_type, extra_prepare_dict)


def deploy(model, config):
    backend_type = BackendType.Academic if not hasattr(
        config.quantize, 'backend') else backend_dict[config.quantize.backend]
    output_path = './' if not hasattr(
        config.quantize, 'deploy') else config.quantize.deploy.output_path
    model_name = config.quantize.deploy.model_name
    deploy_to_qlinear = False if not hasattr(
        config.quantize.deploy, 'deploy_to_qlinear') else config.quantize.deploy.deploy_to_qlinear

    convert_deploy(model, backend_type, {
                   'input': [1, 3, 224, 224]}, output_path=output_path, model_name=model_name, deploy_to_qlinear=deploy_to_qlinear)

def get_quantize(net_fp32, trainloader, qconfig):
    # qconfig = 'fbgemm' or 'qnnpack'
    net_fp32.eval()

    net_fp32.qconfig = torch.quantization.get_default_qconfig(qconfig)
    model_fp32_prepared = torch.quantization.prepare(net_fp32, inplace=False)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        model_fp32_prepared(inputs.to('cpu'))
        break
    model_int8 = torch.quantization.convert(model_fp32_prepared, inplace=False)
    return model_int8

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ImageNet Solver')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--choice', required=True, type=str)
    
    args = parser.parse_args()
    config = parse_config(args.config)


    print(args)
    print(config)
    # seed first
    seed_all(config.process.seed)


    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def import_and_get_model(template_name):
        import sys
        sys.path.append('../../../setting')
        
        template_module = __import__('setting.config', fromlist=[template_name])
        template_function = getattr(template_module, template_name, None)
        
        if template_function is not None:
            return template_function()
        else:
            raise ImportError(f"Function {template_name} not found in template module.")

    # template_name = 'pq_cifar_fp_new'  
    template_name = args.choice
    model, train_loader, val_loader, trainloader_bd, valloader_bd, val_loader_no_targets = import_and_get_model(template_name)




    if hasattr(config, 'quantize'):
        model = get_quantize_model(model, config)
    model.cuda()


    # evaluate
    if not hasattr(config, 'quantize'):
        evaluate(val_loader, model)



    elif config.quantize.quantize_type == 'advanced_ptq':
        print('begin calibration now!')
        dataset_new = train_loader.dataset
        train_loader = torch.utils.data.DataLoader(dataset_new, batch_size=32, shuffle=True, num_workers=16, pin_memory=True)

        cali_data = load_calibrate_data(train_loader, cali_batchsize=config.quantize.cali_batchsize)
        from mqbench.utils.state import enable_quantization, enable_calibration_woquantization
        # do activation and weight calibration seperately for quick MSE per-channel for weight one
        model.eval()


        import torch
        with torch.no_grad():
            enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
            for batch in cali_data:
                model(batch.cuda())
            enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
            model(cali_data[0].cuda())


        print('begin advanced PTQ now!')
        if hasattr(config.quantize, 'reconstruction'):
                model = ptq_reconstruction(
                model, cali_data, config.quantize.reconstruction)
        enable_quantization(model)


        print("after quantization")
        print("cda")
        evaluate(val_loader, model)
        print("asr")
        evaluate(valloader_bd, model)
        print("asr_no_targets")
        evaluate(val_loader_no_targets, model)


        if hasattr(config.quantize, 'deploy'):
            deploy(model, config)


    elif config.quantize.quantize_type == 'naive_ptq':
        print('begin calibration now!')

        cali_data = load_calibrate_data(train_loader, cali_batchsize=config.quantize.cali_batchsize)
        from mqbench.utils.state import enable_quantization, enable_calibration_woquantization
        # do activation and weight calibration seperately for quick MSE per-channel for weight one
        model.eval()
        enable_calibration_woquantization(model, quantizer_type='act_fake_quant')
        for batch in cali_data:
            model(batch.cuda())
        enable_calibration_woquantization(model, quantizer_type='weight_fake_quant')
        model(cali_data[0].cuda())
        print('begin quantization now!')
        enable_quantization(model)




        print("cda")
        evaluate(val_loader, model)
        print("asr")
        evaluate(valloader_bd, model)
        print("asr_no_targets")
        evaluate(val_loader_no_targets, model)


   
        if hasattr(config.quantize, 'deploy'):
            deploy(model, config)
    else:
        print("The quantize_type must in 'naive_ptq' or 'advanced_ptq',")
        print("and 'advanced_ptq' need reconstruction configration.")




