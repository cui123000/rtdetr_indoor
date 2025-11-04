```
# RT-DETR-L

Input: 640x640x3
└─ P1/2  13x13  C=32
   └─ HGStem
└─ P2/4  0x0  C=48
   └─ HGBlock
└─ P3/8  0x0  C=48
   └─ HGBlock
└─ P4/16  0x0  C=48
   └─ HGBlock
└─ P5/32  0x0  C=48
   └─ HGBlock
└─ P?/64  0x0  C=48
   └─ HGBlock
└─ P?/128  0x0  C=48
   └─ HGBlock
└─ P?/256  0x0  C=128
   └─ DWConv
└─ P?/512  0x0  C=96
   └─ HGBlock
└─ P?/1024  0x0  C=96
   └─ HGBlock
└─ P?/2048  0x0  C=96
   └─ HGBlock
└─ P?/4096  0x0  C=96
   └─ HGBlock
└─ P?/8192  0x0  C=96
   └─ HGBlock
└─ P?/16384  0x0  C=96
   └─ HGBlock
└─ P?/32768  0x0  C=512
   └─ DWConv
└─ P?/65536  0x0  C=192
   └─ HGBlock
└─ P?/131072  0x0  C=192
   └─ HGBlock
└─ P?/262144  0x0  C=192
   └─ HGBlock
└─ P?/524288  0x0  C=192
   └─ HGBlock
└─ P?/1048576  0x0  C=192
   └─ HGBlock
└─ P?/2097152  0x0  C=192
   └─ HGBlock
└─ P?/4194304  0x0  C=192
   └─ HGBlock
└─ P?/8388608  0x0  C=192
   └─ HGBlock
└─ P?/16777216  0x0  C=192
   └─ HGBlock
└─ P?/33554432  0x0  C=192
   └─ HGBlock
└─ P?/67108864  0x0  C=192
   └─ HGBlock
└─ P?/134217728  0x0  C=192
   └─ HGBlock
└─ P?/268435456  0x0  C=192
   └─ HGBlock
└─ P?/536870912  0x0  C=192
   └─ HGBlock
└─ P?/1073741824  0x0  C=192
   └─ HGBlock
└─ P?/2147483648  0x0  C=192
   └─ HGBlock
└─ P?/4294967296  0x0  C=192
   └─ HGBlock
└─ P?/8589934592  0x0  C=192
   └─ HGBlock
└─ P?/17179869184  0x0  C=1024
   └─ DWConv
└─ P?/34359738368  0x0  C=384
   └─ HGBlock
└─ P?/68719476736  0x0  C=384
   └─ HGBlock
└─ P?/137438953472  0x0  C=384
   └─ HGBlock
└─ P?/274877906944  0x0  C=384
   └─ HGBlock
└─ P?/549755813888  0x0  C=384
   └─ HGBlock
└─ P?/1099511627776  0x0  C=384
   └─ HGBlock

Head: AIFI + FPN/PAN + RTDETRDecoder (P3/8, P4/16, P5/32)
```
