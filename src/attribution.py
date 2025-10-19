import matplotlib.pyplot as plt
import numpy as np
from captum.attr import (LRP, DeepLift, GuidedBackprop, GuidedGradCam,
                         InputXGradient, IntegratedGradients, LayerAttribution,
                         LayerGradCam, Saliency)

ATTRIBUTION_METHODS = {
    "saliency": Saliency,
    "input_gradient": InputXGradient,
    "guided_backprop": GuidedBackprop,
    "integrated_gradients": IntegratedGradients,
    "deep_lift": DeepLift,
    "lrp": LRP,
    "gradcam": LayerGradCam,
    "guided_gradcam": GuidedGradCam,
}


class Attribution:
    """
    Apply feature attribution method to a sample (x, y)
    """

    def __init__(self, model, attr_method):
        self.model = model
        self.attr_method = attr_method
        if "gradcam" in attr_method: # Grad-CAM & Guided Grad-CAM
            self.attr_func = ATTRIBUTION_METHODS[attr_method](
                model, model.backbone.layer4
            )
        else:
            self.attr_func = ATTRIBUTION_METHODS[attr_method](model)
        
    def apply(self, x, y) -> np.ndarray:
        if self.attr_method == "saliency":
            attr_x = self.attr_func.attribute(x, target=y, abs=False)
        elif self.attr_method == "guided_gradcam":
            attr_x = self.attr_func.attribute(x, target=y, interpolate_mode="bilinear")
        else:
            attr_x = self.attr_func.attribute(x, target=y)

        # Interpolation for GradCAM
        if self.attr_method == "gradcam":
            attr_x = LayerAttribution.interpolate(attr_x, x.shape[-2:])

        return attr_x.detach().cpu().numpy()


def get_plot_range(min_value, max_value, coff=1):
    baseline_value = (min_value + max_value) / 2
    amplitude = max_value - baseline_value
    plot_range = (baseline_value - amplitude * coff, baseline_value + amplitude * coff)
    return plot_range

def plot_attribution(x, y, prob, attr_x, attr_method=None, path=None):
    ATTR_FIGSIZE = (40, 7)
    ECG_COLOR = "darkblue"
    ECG_LW = 2
    ECG_ALPHA = 0.8
    ATTR_COLOR = "crimson"
    ATTR_ALPHA = 0.4
    ATTR_LW = 4

    fig, ax1 = plt.subplots(figsize=ATTR_FIGSIZE)
    ax2 = ax1.twinx()

    # ECG
    ecg_yrange = get_plot_range(np.min(x), np.max(x), 1.35)
    ax1.set_ylim(*ecg_yrange)
    ax1.plot(x.squeeze(), c=ECG_COLOR, linewidth=ECG_LW, alpha=ECG_ALPHA)
    
    # ax1.grid(which="major", axis="x", linestyle="--")
    # ax1.set_ylabel("ECG signal", color=ECG_COLOR)
    ax1.set_xticks(ticks=[250*5*i for i in range(12)], labels=[5*i for i in range(12)], fontdict={"size": 16})
    ax1.set_xlabel("Time (seconds)", fontdict={"size": 18})
    # ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    # Attribution
    if attr_x is not None:
        max_abs_attr = np.max(np.abs(attr_x))
        if np.any(attr_x < 0):
            attr_yrange = (-max_abs_attr * 1.5, max_abs_attr * 1.5)
        else:
            attr_yrange = (-max_abs_attr * 0.07, max_abs_attr * 1.5)
        ax2.set_ylim(*attr_yrange)
        ax2.plot(attr_x.squeeze(), c=ATTR_COLOR, alpha=ATTR_ALPHA, linewidth=ATTR_LW)
        # ax2.set_ylabel("Attribution value", color=ATTR_COLOR)
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)

    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.set_frame_on(False)
    ax1.margins(x=0)
    ax2.margins(x=0)
    
    ax1.set_title(f"label: {y}, prob: {prob:.4f}", fontdict={"size": 18})
    
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=300)
    else:
        plt.show()
    plt.close()