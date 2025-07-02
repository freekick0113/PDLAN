import torch
from mmdet.core import bbox2roi, build_assigner, build_sampler
from mmdet.models import HEADS, build_head, build_roi_extractor
from mmdet.models.roi_heads import StandardRoIHead
import numpy as np
import mmcv

@HEADS.register_module()
class QuasiDenseSegRoIHeadRefineAll(StandardRoIHead):

    def __init__(self,
                 track_roi_extractor=None,
                 track_head=None,
                 track_train_cfg=None,
                 refine_head=None,
                 double_train=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert track_head is not None
        self.track_train_cfg = track_train_cfg
        self.init_track_head(track_roi_extractor, track_head)
        if self.track_train_cfg:
            self.init_track_assigner_sampler()
        assert self.mask_head is not None
        assert refine_head is not None
        self.init_refine_head(refine_head)
        self.double_train = double_train

    def init_track_assigner_sampler(self):
        """Initialize assigner and sampler."""
        if self.track_train_cfg.get('assigner', None):
            self.track_roi_assigner = build_assigner(
                self.track_train_cfg.assigner)
            self.track_share_assigner = False
        else:
            self.track_roi_assigner = self.bbox_assigner
            self.track_share_assigner = True

        if self.track_train_cfg.get('sampler', None):
            self.track_roi_sampler = build_sampler(
                self.track_train_cfg.sampler, context=self)
            self.track_share_sampler = False
        else:
            self.track_roi_sampler = self.bbox_sampler
            self.track_share_sampler = True

    @property
    def with_track(self):
        """bool: whether the RoI head contains a `track_head`"""
        return hasattr(self, 'track_head') and self.track_head is not None

    @property
    def with_refine(self):
        return hasattr(self, 'refine_head') and self.refine_head is not None

    def init_refine_head(self, refine_head):
        self.refine_head = build_head(refine_head)

    def init_weights(self, *args, **kwargs):
        super().init_weights(*args, **kwargs)
        self.refine_head.init_weights()

    def init_track_head(self, track_roi_extractor, track_head):
        """Initialize ``track_head``"""
        if track_roi_extractor is not None:
            self.track_roi_extractor = build_roi_extractor(track_roi_extractor)
            self.track_share_extractor = False
        else:
            self.track_share_extractor = True
            self.track_roi_extractor = self.bbox_roi_extractor
        self.track_head = build_head(track_head)

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            pos_rois = bbox2roi([res.pos_bboxes for res in sampling_results])
            if pos_rois.shape[0] == 0:
                return dict(loss_mask=None)
            mask_results = self._mask_forward(x, pos_rois)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)
            if pos_inds.shape[0] == 0:
                return dict(loss_mask=None)
            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)
        mask_results.update(loss_mask=loss_mask, mask_targets=mask_targets)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)

        return mask_results

    def forward_train(self,
                      x, 
                      img_metas,
                      proposal_list, 
                      gt_bboxes, 
                      gt_labels, 
                      gt_match_indices,
                      # defines the gt box at current image matching relation with the boxes in the reference image, unmatch given -1
                      ref_x, 
                      ref_img_metas, 
                      ref_proposals, 
                      ref_gt_bboxes, 
                      ref_gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      ref_gt_bboxes_ignore=None,
                      ref_gt_masks=None,
                      *args,
                      **kwargs):

        num_imgs = len(img_metas) 
        losses = dict()
        if gt_bboxes_ignore is None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)] 

        key_sampling_results = [] 
        for i in range(num_imgs):
            key_assign_result = self.bbox_assigner.assign(
                proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                gt_labels[i])

            key_sampling_result = self.bbox_sampler.sample(
                key_assign_result,
                proposal_list[i],
                gt_bboxes[i],
                gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in x])

            key_sampling_results.append(key_sampling_result)

        key_mask_results = self._mask_forward_train(
            x, key_sampling_results, None, gt_masks, img_metas)

        if ref_gt_bboxes_ignore is None:
            ref_gt_bboxes_ignore = [None for _ in range(num_imgs)]
        ref_sampling_results = []
        for i in range(num_imgs):
            ref_assign_result = self.bbox_assigner.assign(
                ref_proposals[i], ref_gt_bboxes[i], ref_gt_bboxes_ignore[i],
                ref_gt_labels[i])
            ref_sampling_result = self.bbox_sampler.sample(
                ref_assign_result,
                ref_proposals[i],
                ref_gt_bboxes[i],
                ref_gt_labels[i],
                feats=[lvl_feat[i][None] for lvl_feat in ref_x])
            ref_sampling_results.append(ref_sampling_result)

        ref_mask_results = self._mask_forward_train(
            ref_x, ref_sampling_results, None, ref_gt_masks, ref_img_metas)

        refine_results = self._refine_forward_train(
            key_sampling_results, ref_sampling_results, key_mask_results,
            ref_mask_results, x, ref_x, gt_match_indices)
        if refine_results['loss_refine'] is not None:
            losses.update(loss_refine=refine_results['loss_refine'])

        return losses

    def _track_forward(self, x, bboxes):
        """Track head forward function used in both training and testing."""
        rois = bbox2roi(bboxes)
        track_feats = self.track_roi_extractor(
            x[:self.track_roi_extractor.num_inputs], rois)
        track_feats = self.track_head(track_feats)
        return track_feats

    def simple_test(self, x, img_metas, proposal_list, rescale):
        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        # TODO: support batch inference
        det_bboxes = det_bboxes[0]
        det_labels = det_labels[0]

        det_masks = self.simple_test_mask(
            x, img_metas, det_bboxes, det_labels, rescale=rescale)

        if det_bboxes.size(0) == 0:
            return det_bboxes, det_labels, det_masks, None

        track_bboxes = det_bboxes[:, :-1] * torch.tensor(
            img_metas[0]['scale_factor']).to(det_bboxes.device)
        track_feats = self._track_forward(x, [track_bboxes])

        return det_bboxes, det_labels, det_masks, track_feats

    def simple_test_mask(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        """Simple test for mask head without augmentation."""
        # image shape of the first image in the batch (only one)
        ori_shaep = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            mask_results = dict(mask_pred=None, mask_feats=None)
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(
                    det_bboxes.device)
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_results = self._mask_forward(x, mask_rois)
        return mask_results

    def get_seg_masks(self, img_metas, det_bboxes, det_labels, det_masks,
                      rescale=False):
        """Simple test for mask head without augmentation."""
        # image shape of the first image in the batch (only one)
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes)]
            det_segms = []
            labels = []
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(
                    det_bboxes.device)
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            segm_result, det_segms, labels = self.mask_head.get_seg_masks(
                det_masks['mask_pred'], _bboxes, det_labels, self.test_cfg,
                ori_shape, scale_factor, rescale)
        return segm_result, det_segms, labels

    def _refine_forward(self, key_feats, key_masks, key_labels, ref_feats,
                        ref_masks):
        num_rois = key_masks.size(0)
        inds = torch.arange(0, num_rois, device=key_masks.device).long()
        key_masks = key_masks[inds, key_labels].unsqueeze(dim=1)
        ref_masks = ref_masks[inds, key_labels].unsqueeze(dim=1)

        if self.double_train and self.training:
            ref_masks = self.refine_head(ref_feats, ref_masks, ref_feats,
                                         ref_masks).detach()

        refine_pred = self.refine_head(ref_feats, ref_masks, key_feats,
                                       key_masks)
        refine_results = dict(refine_pred=refine_pred)
        return refine_results

    def _refine_forward_train(self, key_sampling_results, ref_sampling_results,
                              key_mask_results, ref_mask_results, x, ref_x,
                              gt_match_inds):

        num_key_rois = [len(res.pos_bboxes) for res in key_sampling_results] 
        key_pos_pids = [
            gt_match_ind[res.pos_assigned_gt_inds]
            for res, gt_match_ind in zip(key_sampling_results, gt_match_inds)]
        key_pos_bboxes = [res.pos_bboxes for res in key_sampling_results] 
        key_embeds = torch.split(
            self._track_forward(x, key_pos_bboxes), num_key_rois) 

        num_ref_rois = [len(res.pos_bboxes) for res in ref_sampling_results] 
        ref_pos_pids = [
            res.pos_assigned_gt_inds for res in ref_sampling_results]
        ref_pos_bboxes = [res.pos_bboxes for res in ref_sampling_results] 
        ref_embeds = torch.split(
            self._track_forward(ref_x, ref_pos_bboxes), num_ref_rois)

        valids, ref_inds = self.refine_head.match(
            key_embeds, ref_embeds, key_pos_pids, ref_pos_pids)

        def valid_select(inputs, num_splits, inds):
            inputs = torch.split(inputs, num_splits)
            inputs = torch.cat(
                [input_[ind] for input_, ind in zip(inputs, inds)])
            return inputs

        key_feats = valid_select(
            key_mask_results['mask_feats'], num_key_rois, valids)
        key_masks = valid_select(
            key_mask_results['mask_pred'], num_key_rois, valids)
        key_targets = valid_select(
            key_mask_results['mask_targets'], num_key_rois, valids)
        key_labels = torch.cat(
            [res.pos_gt_labels[valid]
            for res, valid in zip(key_sampling_results, valids)])
        ref_feats = valid_select(
            ref_mask_results['mask_feats'], num_ref_rois, ref_inds)
        ref_masks = valid_select(
            ref_mask_results['mask_pred'], num_ref_rois, ref_inds)


        if key_masks.size(0) == 0:
            key_feats = key_mask_results['mask_feats']
            key_masks = key_mask_results['mask_pred']
            key_targets = key_mask_results['mask_targets']
            key_labels = torch.cat([
                res.pos_gt_labels for res in key_sampling_results])
            ref_feats = key_feats.detach()
            ref_masks = key_masks.detach()

        refine_results = self._refine_forward(key_feats, key_masks, key_labels,
                                              ref_feats, ref_masks)
        refine_targets = key_targets
        loss_refine = self.refine_head.loss_mask(
            refine_results['refine_pred'].squeeze(dim=1), refine_targets)
        refine_results.update(loss_refine=loss_refine,
                              refine_targets=refine_targets)
        return refine_results

    def simple_test_refine(self, img_metas, key_feats, key_masks, key_bboxes,
                           key_labels, ref_feats, ref_masks, rescale=False):
        # print("key_masks:",key_masks)
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        if key_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes)]
            det_segms = []
            refine_preds = key_masks.new_full(key_masks.size())
            print("key_bboxes.shape[0] == 0")
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = torch.from_numpy(scale_factor).to(
                    key_bboxes.device)
            _bboxes = (
                key_bboxes[:, :4] * scale_factor if rescale else key_bboxes)
            refine_results = self._refine_forward(
                key_feats, key_masks, key_labels, ref_feats, ref_masks)


            key_masks_cls = key_masks[range(len(key_masks)), key_labels, :, :].unsqueeze(1)
            select_inds = torch.where(key_labels >= 10)
            refine_preds = refine_results['refine_pred']
            refine_preds[select_inds] = key_masks_cls[select_inds]


            segm_result, det_segms = self.refine_head.get_seg_masks(
                refine_preds, _bboxes, key_labels, self.test_cfg, ori_shape,
                scale_factor, rescale)

        return segm_result, det_segms, refine_preds
