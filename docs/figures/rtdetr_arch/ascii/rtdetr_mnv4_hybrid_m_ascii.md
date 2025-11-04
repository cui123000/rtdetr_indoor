```
# RT-DETR-MNV4-Hybrid-M

Input: 640x640x3
└─ P1/2  320x320  C=32
   └─ Conv
└─ P2/4  160x160  C=48
   └─ EdgeResidual
└─ P3/8  80x80  C=80
   └─ UniversalInvertedResidual
   └─ UniversalInvertedResidual
└─ P4/16  40x40  C=160
   ├─ UniversalInvertedResidual
   ├─ UniversalInvertedResidual
   ├─ UniversalInvertedResidual
   ├─ UniversalInvertedResidual
   └─ C2f
   ├─ UniversalInvertedResidual
   └─ C2f
   ├─ UniversalInvertedResidual
   └─ C2f
└─ P5/32  20x20  C=960
   ├─ UniversalInvertedResidual
   ├─ UniversalInvertedResidual
   ├─ UniversalInvertedResidual
   ├─ UniversalInvertedResidual
   ├─ UniversalInvertedResidual
   ├─ UniversalInvertedResidual
   ├─ UniversalInvertedResidual
   ├─ UniversalInvertedResidual
   ├─ C2f
   ├─ UniversalInvertedResidual
   └─ Conv

Head: AIFI + FPN/PAN + RTDETRDecoder (P3/8, P4/16, P5/32)
```
