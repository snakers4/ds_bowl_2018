from utils.RAugs import *

class BAugs(object):
    def __init__(self,
                 prob=0.5,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.prob = prob
        self.mean = mean
        self.std = std
    def __call__(self, img, mask, target_resl):
        return DualCompose([
            Resize(size=target_resl),
            ImageOnly(CLAHE()),
            OneOrOther(
                *(OneOf([
                    Distort1(distort_limit=0.05, shift_limit=0.05),
                    Distort2(num_steps=2, distort_limit=0.05)]),
                  ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.10, rotate_limit=0)), prob=self.prob),
            RandomFlip(prob=0.5),
            ImageOnly(RandomContrast(limit=0.2, prob=self.prob)),
            ImageOnly(RandomFilter(limit=0.5, prob=self.prob/2)),
            ImageOnly(Normalize(mean=self.mean, std=self.std)),            
        ])(img, mask)
    
class BAugsVal(object):
    def __init__(self,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
    def __call__(self, img, mask, target_resl):
        return DualCompose([
            Resize(size=target_resl),
            ImageOnly(CLAHE()),            
            ImageOnly(Normalize(mean=self.mean, std=self.std)),
        ])(img, mask)    
    
class BAugsNoResize(object):
    def __init__(self,
                 prob=0.5,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.prob = prob
        self.mean = mean
        self.std = std
    def __call__(self, img, mask):
        return DualCompose([
            ImageOnly(CLAHE()),
            OneOrOther(
                *(OneOf([
                    Distort1(distort_limit=0.05, shift_limit=0.05),
                    Distort2(num_steps=2, distort_limit=0.05)]),
                  ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.10, rotate_limit=0)), prob=self.prob),
            RandomFlip(prob=0.5),
            ImageOnly(RandomContrast(limit=0.2, prob=self.prob)),
            ImageOnly(RandomFilter(limit=0.5, prob=self.prob/2)),
            ImageOnly(Normalize(mean=self.mean, std=self.std)),            
        ])(img, mask)
    
class BAugsValNoResize(object):
    def __init__(self,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
    def __call__(self, img, mask):
        return DualCompose([
            # ImageOnly(CLAHE()),
            ImageOnly(AlwaysGray()),            
            ImageOnly(Normalize(mean=self.mean, std=self.std)),
        ])(img, mask)
    
class BAugsValPad(object):
    def __init__(self,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std
    def __call__(self, img, mask):
        return DualCompose([
            # ImageOnly(CLAHE()),
            ImageOnly(AlwaysGray()),            
            ImageOnly(Normalize(mean=self.mean, std=self.std)),
        ])(img, mask)      
    
class BAugsNoResizeCrop(object):
    def __init__(self,
                 prob=0.5,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225)):
        self.prob = prob
        self.mean = mean
        self.std = std
    def __call__(self, img, mask):
        return DualCompose([
            # ImageOnly(CLAHE()),
            OneOrOther(
                *(OneOf([
                    Distort1(distort_limit=0.05, shift_limit=0.05),
                    Distort1(distort_limit=0.05, shift_limit=0.05),
                    # Distort2(num_steps=2, distort_limit=0.05)
                    ]),
                  ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.10, rotate_limit=45)), prob=self.prob),
            RandomFlip(prob=0.5),
            ImageOnly(RandomContrast(limit=0.2, prob=self.prob)),
            ImageOnly(RandomFilter(limit=0.5, prob=self.prob/2)),
            ImageOnly(AlwaysGray()),            
            ImageOnly(Normalize(mean=self.mean, std=self.std)),
            RandomCrop(size=[256,256])
        ])(img, mask)    