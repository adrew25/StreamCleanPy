import cv2
import numpy as np


class MaskPostProcessor:
    def __init__(self, kernel_size_smooth=15, kernel_size_refine=5, alpha_blend=0.5):
        """
        Initialize the MaskPostProcessor with configurable parameters.

        Args:
            kernel_size_smooth (int): Kernel size for Gaussian smoothing.
            kernel_size_refine (int): Kernel size for morphological operations.
            alpha_blend (float): Alpha value for edge blending.
        """
        self.kernel_size_smooth = kernel_size_smooth
        self.kernel_size_refine = kernel_size_refine
        self.alpha_blend = alpha_blend

    def smooth_mask(self, mask):
        """
        Smooth the mask using Gaussian blur.
        """
        smoothed_mask = cv2.GaussianBlur(
            mask, (self.kernel_size_smooth, self.kernel_size_smooth), 0
        )
        return smoothed_mask

    def refine_mask(self, mask):
        """
        Refine the mask using morphological operations.
        """
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.kernel_size_refine, self.kernel_size_refine)
        )
        dilated_mask = cv2.dilate(mask, kernel, iterations=1)
        eroded_mask = cv2.erode(dilated_mask, kernel, iterations=1)
        return eroded_mask

    def blend_edges(self, frame, mask):
        """
        Blend edges of the mask with the background for smoother transitions.
        """
        blurred_frame = cv2.GaussianBlur(frame, (21, 21), 0)
        blended_frame = cv2.addWeighted(
            blurred_frame, self.alpha_blend, frame, 1 - self.alpha_blend, 0
        )
        return np.where(mask[:, :, None] == 255, frame, blended_frame)

    def process_mask(self, frame, mask):
        """
        Full mask post-processing pipeline.
        Applies refine, smooth, and optionally blends edges.

        Args:
            frame (numpy.ndarray): Original video frame.
            mask (numpy.ndarray): Binary mask of detected objects.

        Returns:
            numpy.ndarray: Post-processed mask.
        """
        refined_mask = self.refine_mask(mask)
        smoothed_mask = self.smooth_mask(refined_mask)
        blended_frame = self.blend_edges(frame, smoothed_mask)
        return smoothed_mask, blended_frame
