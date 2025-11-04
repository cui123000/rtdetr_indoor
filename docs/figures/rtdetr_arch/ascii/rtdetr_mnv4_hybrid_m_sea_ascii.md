```
# RT-DETR-MNV4-Hybrid-M-SEA

Input: 640x640x3
└─ P1/2  320x320  C=32
   └─ Conv
└─ P2/4  160x160  C=48
   └─ EdgeResidual
└─ P3/8  80x80  C=80 [SEA]
   ├─ UniversalInvertedResidual
   ├─ UniversalInvertedResidual
   └─ Sea_Attention_Simplified [SEA]
└─ P4/16  40x40  C=160 [SEA]
   ├─ UniversalInvertedResidual
   ├─ UniversalInvertedResidual
   ├─ UniversalInvertedResidual
   ├─ OptimizedSEA_Attention [SEA]
   ├─ UniversalInvertedResidual
   └─ C2f
   ├─ UniversalInvertedResidual
   └─ C2f
   ├─ UniversalInvertedResidual
   ├─ OptimizedSEA_Attention [SEA]
   └─ C2f
└─ P5/32  20x20  C=960 [SEA]
   ├─ UniversalInvertedResidual
   ├─ UniversalInvertedResidual
   ├─ TransformerEnhancedSEA [SEA]
   ├─ UniversalInvertedResidual
   ├─ UniversalInvertedResidual
   ├─ UniversalInvertedResidual
   ├─ UniversalInvertedResidual
   ├─ UniversalInvertedResidual
   ├─ UniversalInvertedResidual
   ├─ TransformerEnhancedSEA [SEA]
   ├─ C2f
   ├─ UniversalInvertedResidual
   └─ Conv

Head: AIFI + FPN/PAN + RTDETRDecoder (P3/8, P4/16, P5/32)
```
