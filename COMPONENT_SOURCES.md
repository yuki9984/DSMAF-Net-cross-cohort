# Sources and local compatibility changes
| Component | Pinned source | Release treatment |
|---|---|---|
| AFPN | gyyang23/AFPN, 0930b563b43830c68825241689b80d8e430f2576 | Apache-2.0; standalone four-output variant |
| Haar DWT | YehLi/ImageNetModel, 0b4302e0b10ff12013b03b799dda50393efa69a9, classification/torch_wavelets.py | Apache-2.0; only forward transform, explicit precision adaptation |
| Position/channel attention | CASIA-IVA-Lab/DANet, b0a0fa008148f102321cdaa7ca38fb0ddd0239d1, encoding/nn/da_att.py | MIT; projections and head adapted to historical checkpoints |
| MSCBLayer | SLDGroup/EMCAD, 26c9c31f731f749b62c5fe83f44376dac75f3aa8, lib/decoders.py | Obtain unchanged source directly; UT Austin research terms |
| MetaBlock | paaatcha/MetaBlock, e283174409f5e5685cb871bbb9e85c26c3a53c65 | Existing MIT adapter and notice retained |
| RG-DermNet | life-ufes/rg-dermnet, 940c1cd3598b5a887aaec118371164dc256d1cb3 | Three original files obtained directly; CC BY-NC 4.0 |
| Unused historical imports | Agent Attention helper, BFAM, B2CNet, HLNet/IDWT | Omitted; disabled branches fail explicitly |
The historical mixed wrapper is replaced by a minimal loader. It retains the
AFPN .impl state-dictionary prefix and refuses missing or altered dependencies.
Original trained weights, stored predictions and analysis plans are unchanged.
The study's metadata-conditioned CrossModalAgentAttention is distinct from
the unused external Agent Attention helper.
These adaptations preserve the evaluated model, and are not additional
architecture innovations. See DEPENDENCY_ACCESS.md and COMPATIBILITY_REPORT.json.
