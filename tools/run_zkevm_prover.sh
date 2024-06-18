#!/bin/sh
RUN_TASK="build/zkProver -c config/config_prover.json"
if [ "$1" = "monitor" ]; then
	PREFIX="*/1 * * * * "
	PREFIX_1="\*/1 \* \* \* \* "
	TASK="cd $PWD;./tools/monitor_restart.sh >> logs/monitor/monitor.log 2>&1"
	if crontab -l | grep "${PREFIX_1}${TASK}"; then
		echo "The task is in the crontab."
	else
		echo "The task is not in the crontab."
		(
			crontab -l 2>/dev/null
			echo "${PREFIX}${TASK}"
		) | crontab -
	fi
	echo "Monitor mode activated."
else
	echo "No monitor mode. Running in default mode."
fi

mkdir -p logs/output
mkdir -p logs/monitor
# shellcheck disable=SC2046
nohup "${RUN_TASK}" >logs/output/$(date +%Y%m%d_%H%M%S).log 2>&1 &
PID=$!

if [ "$1" = "monitor" ]; then

	echo "#!/bin/sh
  # shellcheck disable=SC2009
  CMD=\"ps -ef | grep ${PID} | grep -v \\\"grep\\\"\"
  echo \"Executing command: \$CMD\"
  eval \"\$CMD\"
  if [ \"\$?\" -eq 1 ]
  then
  	bash ./tools/run_zkevm_prover.sh monitor
  	echo \"\$(date) process has been restarted!\"
  else
  	echo \"\$(date) process already started!\"
  fi" >./tools/monitor_restart.sh

	chmod +x ./tools/monitor_restart.sh

fi