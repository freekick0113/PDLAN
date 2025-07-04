import os.path as osp
import shutil
import tempfile
import time
from collections import defaultdict

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmdet.core import encode_mask_results #, tensor2imgs
from mmcv.image import tensor2imgs
from pdlan.core import encode_track_results


def single_gpu_test(model, 
                    data_loader, 
                    show=False, 
                    out_dir=None, 
                    show_score_thr=0.3): 
    model.eval() 
    results = defaultdict(list) 
    dataset = data_loader.dataset 
    prog_bar = mmcv.ProgressBar(len(dataset)) 
    for i, data in enumerate(data_loader): 
        with torch.no_grad(): 
            result = model(return_loss=False, rescale=True, **data)

        if show or out_dir: 
            img_tensor = data['img'][0] 
            img_metas = data['img_metas'][0].data[0] 
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg']) 

            assert len(imgs) == len(img_metas) 

            for img, img_meta in zip(imgs, img_metas):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]
                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))
                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result( 
                    img_show,
                    result,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr,
                    draw_track=True)

        if 'segm_result' in result:
            result['segm_result'] = encode_mask_results(result['segm_result'])
        if 'track_result' in result:
            result['track_result'] = (
                encode_track_results(result['track_result']))
        for k, v in result.items():
            results[k].append(v) 
        batch_size = data['img'][0].size(0) 
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = defaultdict(list)
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
       
        if 'segm_result' in result:
            result['segm_result'] = encode_mask_results(result['segm_result'])
        if 'track_result' in result:
            result['track_result'] = (
                encode_track_results(result['track_result']))
        for k, v in result.items():
            results[k].append(v)
            
        if rank == 0:
            batch_size = (
                len(data['img_meta']._data)
                if 'img_meta' in data else data['img'][0].size(0))
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        raise NotImplementedError
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = defaultdict(list)
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_file = mmcv.load(part_file)
            for k, v in part_file.items():
                part_list[k].extend(v)
        shutil.rmtree(tmpdir)
        return part_list
