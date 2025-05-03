@REM python ./ensemble.py --config-path ./config/evaluate_config/ntu60_cs_best.json

@REM python ./ensemble.py --config-path ./config/evaluate_config/ntu60_cs_last.json

python ./ensemble.py --config-path ./config/evaluate_config/ntu60_cs_loss.json

@REM python ./ensemble.py --config-path ./config/evaluate_config/ntu60_cv_best.json

@REM python ./ensemble.py --config-path ./config/evaluate_config/ntu60_cv_last.json

python ./ensemble.py --config-path ./config/evaluate_config/ntu60_cv_loss.json

@REM python ./ensemble.py --config-path ./config/evaluate_config/ntu120_cs_best.json

@REM python ./ensemble.py --config-path ./config/evaluate_config/ntu120_cs_last.json

python ./ensemble.py --config-path ./config/evaluate_config/ntu120_cs_loss.json

@REM python ./ensemble.py --config-path ./config/evaluate_config/ntu120_cv_best.json

@REM python ./ensemble.py --config-path ./config/evaluate_config/ntu120_cv_last.json

python ./ensemble.py --config-path ./config/evaluate_config/ntu120_cv_loss.json

@REM shutdown /s /t 60
