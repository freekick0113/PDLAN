import random

import mmcv
import numpy as np
from mmdet.datasets import DATASETS, CocoDataset
from pdlan.core import eval_mot, eval_mots

from .parsers import CocoVID


@DATASETS.register_module()
class CocoVideoDataset(CocoDataset):

    CLASSES = None

    def __init__(self,
                 load_as_video=True,
                 match_gts=True,
                 skip_nomatch_pairs=True,
                 key_img_sampler=dict(interval=1),
                 ref_img_sampler=dict(
                     scope=3, num_ref_imgs=1, method='uniform'),
                 *args,
                 **kwargs):
        self.load_as_video = load_as_video
        self.match_gts = match_gts
        self.skip_nomatch_pairs = skip_nomatch_pairs
        self.key_img_sampler = key_img_sampler
        self.ref_img_sampler = ref_img_sampler
        # self.data_infos = self.load_annotations(self.ann_file)
        super().__init__(*args, **kwargs)


    def load_annotations(self, ann_file):
        """Load annotation from annotation file."""
        if not self.load_as_video:
            data_infos = super().load_annotations(ann_file)
        else:
            data_infos = self.load_video_anns(ann_file)
        self.data_infos = data_infos
        return data_infos

    def load_video_anns(self, ann_file):
        self.coco = CocoVID(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        data_infos = []
        self.vid_ids = self.coco.get_vid_ids()
        self.img_ids = []
        for vid_id in self.vid_ids:
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            img_ids = self.key_img_sampling(img_ids, **self.key_img_sampler)
            self.img_ids.extend(img_ids)
            for img_id in img_ids:
                info = self.coco.load_imgs([img_id])[0]
                if len(info['file_name'].split('/')) > 2:
                    replace_token = info['file_name'].split('/')[0] + '/' + info['file_name'].split('/')[1] + '/'
                    info['file_name'] = info['file_name'].replace(replace_token, info['file_name'].split('/')[0] + '/')
                info['filename'] = info['file_name']
                data_infos.append(info)
        return data_infos

    def key_img_sampling(self, img_ids, interval=1):
        return img_ids[::interval]

    def ref_img_sampling(self,
                         img_info,
                         scope,
                         num_ref_imgs=1,
                         method='uniform'):
        if num_ref_imgs != 1 or method != 'uniform':
            raise NotImplementedError
        if img_info.get('frame_id', -1) < 0 or scope <= 0:
            ref_img_info = img_info.copy()
        else:
            vid_id = img_info['video_id']
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            frame_id = img_info['frame_id']
            if method == 'uniform':
                left = max(0, frame_id - scope)
                right = min(frame_id + scope, len(img_ids) - 1)
                valid_inds = img_ids[left:frame_id] + img_ids[frame_id +
                                                              1:right + 1]
                ref_img_id = random.choice(valid_inds)
            ref_img_info = self.coco.loadImgs([ref_img_id])[0]
            ref_img_info['filename'] = ref_img_info['file_name']
        return ref_img_info

    def ref_img_sampling_test(self,
                         img_info,
                         num_ref_imgs=1,
                         method='uniform'):
        if num_ref_imgs != 1 or method != 'uniform':
            raise NotImplementedError

        if img_info.get('frame_id', -1) <= 0:
            ref_img_info = img_info.copy()
        else:
            vid_id = img_info['video_id']
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            frame_id = img_info['frame_id']
            '''
            if method == 'uniform':
                left = max(0, frame_id - scope)
                right = min(frame_id + scope, len(img_ids) - 1)
                valid_inds = img_ids[left:frame_id] + img_ids[frame_id +
                                                              1:right + 1]
                ref_img_id = random.choice(valid_inds)
            '''
            ref_img_id = img_ids[frame_id - 1]
            ref_img_info = self.coco.loadImgs([ref_img_id])[0]
            ref_img_info['filename'] = ref_img_info['file_name']
        return ref_img_info

    def _pre_pipeline(self, _results):
        super().pre_pipeline(_results)
        _results['frame_id'] = _results['img_info'].get('frame_id', -1)

    def pre_pipeline(self, results):
        """Prepare results dict for pipeline."""
        if isinstance(results, list):
            for _results in results:
                self._pre_pipeline(_results)
        elif isinstance(results, dict):
            self._pre_pipeline(results)
        else:
            raise TypeError('input must be a list or a dict')

    def get_ann_info(self, img_info):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        img_id = img_info['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id], cat_ids=self.cat_ids)
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(img_info, ann_info)

    def prepare_results(self, img_info):
        ann_info = self.get_ann_info(img_info)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            idx = self.img_ids.index(img_info['id'])
            results['proposals'] = self.proposals[idx]
        return results

    def match_results(self, results, ref_results):
        match_indices, ref_match_indices = self._match_gts(
            results['ann_info'], ref_results['ann_info'])
        results['ann_info']['match_indices'] = match_indices
        ref_results['ann_info']['match_indices'] = ref_match_indices
        return results, ref_results

    def _match_gts(self, ann, ref_ann):
        if 'instance_ids' in ann:
            ins_ids = list(ann['instance_ids'])
            ref_ins_ids = list(ref_ann['instance_ids'])
            match_indices = np.array([
                ref_ins_ids.index(i) if i in ref_ins_ids else -1
                for i in ins_ids
            ])
            ref_match_indices = np.array([
                ins_ids.index(i) if i in ins_ids else -1 for i in ref_ins_ids
            ])
        else:
            match_indices = np.arange(ann['bboxes'].shape[0], dtype=np.int64)
            ref_match_indices = match_indices.copy()
        return match_indices, ref_match_indices

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys \
                introduced by pipeline.
        """
        img_info = self.data_infos[idx]
        #print('img info:', img_info)
        ref_img_info = self.ref_img_sampling(img_info, **self.ref_img_sampler)

        results = self.prepare_results(img_info)
        ref_results = self.prepare_results(ref_img_info)

        if self.match_gts:
            results, ref_results = self.match_results(results, ref_results)
            nomatch = (results['ann_info']['match_indices'] == -1).all()
            if self.skip_nomatch_pairs and nomatch:
                return None

        self.pre_pipeline([results, ref_results])
        return self.pipeline([results, ref_results])
    
    # def prepare_test_img(self, idx):
    #     """Get training data and annotations after pipeline.

    #     Args:
    #         idx (int): Index of data.

    #     Returns:
    #         dict: Training data and annotation after pipeline with new keys \
    #             introduced by pipeline.
    #     """
    #     img_info = self.data_infos[idx]
    #     results = dict(img_info=img_info)
    #     ref_img_info = self.ref_img_sampling_test(img_info)
    #     ref_results = dict(img_info=ref_img_info)

    #     self.pre_pipeline([results, ref_results])
    #     return self.pipeline([results, ref_results])
    
    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        gt_instance_ids = []
        dict_label={}

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                if self.cat2label[ann['category_id']] in dict_label.keys():
                    dict_label[self.cat2label[ann['category_id']]] +=1
                else:
                    dict_label[self.cat2label[ann['category_id']]] =0
                # if self.cat2label[ann['category_id']] not in [0,1]:
                #     print(self.cat2label[ann['category_id']])
                # print(self.cat2label[ann['category_id']])
                if ann.get('segmentation', False):
                    gt_masks_ann.append(ann['segmentation'])
                instance_id = ann.get('instance_id', None)
                if instance_id is not None:
                    gt_instance_ids.append(ann['instance_id'])
        # print(dict_label.keys())   
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        if self.load_as_video:
            ann['instance_ids'] = np.array(gt_instance_ids).astype(np.int_)
        else:
            ann['instance_ids'] = np.arange(len(gt_labels))

        return ann

    def format_track_results(self, results, **kwargs):
        pass

    def evaluate(self,
                 results,
                 metric=['bbox', 'track'],
                 logger=None,
                 classwise=True,
                 mot_class_average=True,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=None,
                 metric_items=None):
        # evaluate for detectors without tracker
        #mot_class_average=False
        mot_class_average=True
        eval_results = dict()
        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'track', 'segtrack']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        ## mAP 
        super_metrics = ['bbox', 'segm']
        if super_metrics:
            if 'bbox' in super_metrics and 'segm' in super_metrics:
                super_results = []
                for bbox, segm in zip(results['bbox_result'], results['segm_result']):
                    super_results.append((bbox, segm))
            else:
                super_results = results['bbox_result']
            super_eval_results = super().evaluate(
                results=super_results,
                metric=super_metrics,
                logger=logger,
                classwise=classwise,
                proposal_nums=proposal_nums,
                iou_thrs=iou_thr,
                metric_items=metric_items)
            eval_results.update(super_eval_results)
        ##mAP-end

        if 'segtrack' in metrics:
            track_eval_results = eval_mots(
                self.coco,
                mmcv.load(self.ann_file),
                results['track_result'],
                class_average=mot_class_average)
            eval_results.update(track_eval_results)
        elif 'track' in metrics:
            track_eval_results = eval_mot(
                mmcv.load(self.ann_file),
                results['track_result'],
                class_average=mot_class_average)
            eval_results.update(track_eval_results)

        return eval_results
