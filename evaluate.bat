@REM python ./ensemble.py --config-path ./config/evaluate_config/ntu60_cs_best.json

@REM python ./ensemble.py --config-path ./config/evaluate_config/ntu60_cs_last.json

python ./ensemble.py --config-path ./config/evaluate_config/ntu60_cs_loss.json

@REM python ./ensemble.py --config-path ./config/evaluate_config/ntu60_cv_best.json

@REM python ./ensemble.py --config-path ./config/evaluate_config/ntu60_cv_last.json

python ./ensemble.py --config-path ./config/evaluate_config/ntu60_cv_loss.json

@REM python ./ensemble.py --config-path ./config/evaluate_config/ntu120_csub_best.json

@REM python ./ensemble.py --config-path ./config/evaluate_config/ntu120_csub_last.json

python ./ensemble.py --config-path ./config/evaluate_config/ntu120_csub_loss.json

@REM python ./ensemble.py --config-path ./config/evaluate_config/ntu120_csubet_best.json

@REM python ./ensemble.py --config-path ./config/evaluate_config/ntu120_csubet_last.json

python ./ensemble.py --config-path ./config/evaluate_config/ntu120_csubet_loss.json

@REM shutdown /s /t 60
