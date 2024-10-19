from pstp.pstp_collections import pstp_scan_saps_prediction,pstp_scan_pdps_prediction,pstp_scan_mix_prediction



test_seq = 'GRGDSPYS'*25
saps_residue_level_scores, saps_scan_predicted, saps_scan_kernel_predicted = pstp_scan_saps_prediction(
    test_seq)
pdps_residue_level_scores, pdps_scan_predicted, pdps_scan_kernel_predicted = pstp_scan_pdps_prediction(
    test_seq)
mix_residue_level_scores, mix_scan_predicted, mix_scan_kernel_predicted = pstp_scan_mix_prediction(
    test_seq)

