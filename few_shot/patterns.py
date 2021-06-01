import pathlib

from numpy.lib.twodim_base import mask_indices

ROOT = pathlib.Path(__file__).parent.parent.absolute()


# Mask Patterns
P1 = "So, the <mask> is the interesting aspect."
P2 = "So, the interesting aspect is <mask>."
P3 = "So, the <mask> are the interesting aspect."
P4 = "So, this is my opinion on <mask>."
P5 = "So, my review focuses on the <mask>."
P6 = "So, the <mask> is wonderful."
P7 = "So, the <mask> is awful."
P8 = "So, the main topic is the <mask>."
P9 = "So, it is all about the <mask>."
P10 = "So, I am talking about the <mask>."

PATTERNS = {'P1': P1, 'P2': P2, 'P3': P3, 'P4': P4, 'P5': P5, 'P6': P6, 'P7': P7, 'P8': P8, 'P9': P9, 'P10': P10}

P_B1 = "So the <aspect> was <mask>."
P_B2 = "In summary, the <aspect> was <mask>."
P_B3 = "All in all, the <aspect> was <mask>."
P_B4 = "<mask>, the aspect is <aspect>."
P_B5 = "<mask>, the aspect in my review is <aspect>."
P_B6 = "<mask>, the topic of my review is <aspect>."
P_B7 = "Is it true that the aspect is <aspect>? <mask>."
P_B8 = "The sentiment towards <aspect> is <mask>."
P_B9 = "The opinion towards <aspect> is <mask>."
P_B10 ="So my review towards the <aspect> is <mask>"
P_B11 ="Is <aspect> the aspect in the previous sentence? <mask>"

P_B12 ="Is there sentiment towards <aspect> in the previous sentence? <mask>"
P_B13 ="So, does the review in the previous sentence focuses on <aspect>? <mask>"
P_B14 ="So, is <aspect> the topic of my review? <mask>"
P_B15 ="does this review focuses on <aspect>? <mask>"
P_B16 ="So, is <aspect> the aspect in the previous sentence? <mask>"


SCORING_PATTERNS = {'P_B1': P_B1, 'P_B2': P_B2, 'P_B3': P_B3, 'P_B4': P_B4, 'P_B5': P_B5, 'P_B6': P_B6, 'P_B7': P_B7, 'P_B8': P_B8, 'P_B9': P_B9,
                    'P_B10': P_B10, 'P_B11': P_B11, 'P_B12': P_B12, 'P_B13': P_B13, 'P_B14': P_B14, 'P_B15': P_B15}
