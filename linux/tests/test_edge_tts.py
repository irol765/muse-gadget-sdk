"""WK-230: Edge backup-route token must match the reference scheme."""

from musegadget.edge_tts import generate_sec_ms_gec_v2


def test_gec_v2_known_vectors():
    # Vectors produced by the reference edge-tts implementation.
    assert generate_sec_ms_gec_v2(1700000000.0) == (
        "2CD23A2CA0B2595A-"
        "F3C846C370E05F9CC83ACC5F995C5F5D87A4D0F14CE8936A0F902E64099E38A2")
    assert generate_sec_ms_gec_v2(1762400000.0) == (
        "5EA6F3526E455995-"
        "D27609FD7F53D2DD938FD2DC52AAAF40B8DEE3B9E5E9EC29DD7CFC02E7E09E6A")
