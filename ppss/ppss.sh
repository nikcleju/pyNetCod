#!/usr/bin/env bash
#
# PPSS, the Parallel Processing Shell Script
# 
# Copyright (c) 2009, Louwrentius
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY Louwrentius ''AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL Louwrentius BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#------------------------------------------------------------------------------
# It should not be necessary to edit antyhing in this script..
# Ofcource you can if it is necessary for your needs.
# Send a patch if your changes may benefit others.
#------------------------------------------------------------------------------

# Handling control-c for a clean shutdown.
trap 'kill_process; ' INT

# Setting some vars. 
SCRIPT_NAME="Distributed Parallel Processing Shell Script"
SCRIPT_VERSION="2.41"

# The first argument to this script can be a mode.
MODES="node start config stop pause continue deploy status erase kill"
for x in $MODES
do
    if [ "$x" == "$1" ]
    then
        MODE="$1"
        shift
    fi
done

# The working directory of PPSS can be set with
# export PPSS_DIR=/path/to/workingdir
if [ -z "$PPSS_DIR" ]
then
    PPSS_DIR="ppss"
fi

if [ ! -e "$PPSS_DIR" ]
then
    mkdir -p "$PPSS_DIR"
fi

CONFIG=""
HOSTNAME=`hostname`
ARCH=`uname`

PID="$$"
GLOBAL_LOCK="$PPSS_DIR/PPSS-GLOBAL-LOCK-$PID"           # Global lock file used by local PPSS instance.
PAUSE_SIGNAL="$PPSS_DIR/pause_signal"                   # Pause processing if this file is present.
PAUSE_DELAY=60                                          # Polling every 1 minutes by default.
STOP_SIGNAL="$PPSS_DIR/stop_signal"                     # Stop processing if this file is present.
ARRAY_POINTER_FILE="$PPSS_DIR/ppss-array-pointer-$PID"  # Pointer for keeping track of processed items.
JOB_LOG_DIR="$PPSS_DIR/job_log"                         # Directory containing log files of processed items.
LOGFILE="$PPSS_DIR/ppss-log-$$.txt"                     # General PPSS log file. Contains lots of info.
STOP=0                                                  # STOP job.
MAX_DELAY=0                                             # MAX DELAY between jobs.
MAX_LOCK_DELAY=3                                        #
PERCENT="0"
LISTENER_PID=""
IFS_BACKUP="$IFS"
CPUINFO=/proc/cpuinfo
PROCESSORS=""
STOP_KEY=$RANDOM$RANDOM$RANDOM
KILL_KEY=$RANDOM$RANDOM$RANDOM

SSH_SERVER=""                           # Remote server or 'master'.
SSH_KEY=""                              # SSH key for ssh account.
SSH_KNOWN_HOSTS=""
SSH_SOCKET="$PPSS_DIR/PPSS_SSH_SOCKET"          # Multiplex multiple SSH connections over 1 master.
SSH_OPTS="-o BatchMode=yes -o ControlPath=$SSH_SOCKET \
                           -o GlobalKnownHostsFile=./known_hosts \
                           -o ControlMaster=auto \
                           -o Cipher=blowfish \
                           -o ConnectTimeout=15 "

                                        # Blowfish is faster but still secure. 
SSH_MASTER_PID=""

PPSS_HOME_DIR=ppss
ITEM_LOCK_DIR="$PPSS_DIR/PPSS_ITEM_LOCK_DIR"      # Remote directory on master used for item locking.
PPSS_LOCAL_TMPDIR="$PPSS_DIR/PPSS_LOCAL_TMPDIR"   # Local directory on slave for local processing.
PPSS_LOCAL_OUTPUT="$PPSS_DIR/PPSS_LOCAL_OUTPUT"   # Local directory on slave for local output.
TRANSFER_TO_SLAVE="0"                   # Transfer item to slave via (s)cp.
SECURE_COPY="1"                         # If set, use SCP, Otherwise, use cp.
REMOTE_OUTPUT_DIR=""                    # Remote directory to which output must be uploaded.
SCRIPT=""                               # Custom user script that is executed by ppss.
ITEM_ESCAPED=""
NODE_STATUS="$PPSS_DIR/status.txt"

showusage_short () {

    echo
    echo "|P|P|S|S| $SCRIPT_NAME $SCRIPT_VERSION"
    echo
    echo "usage: $0 [ -d <sourcedir> | -f <sourcefile> ]  [ -c '<command> \"\$ITEM\"' ]"
    echo "                 [ -C <configfile> ]  [ -j ] [ -l <logfile> ] [ -p <# jobs> ]"
    echo "                 [ -D <delay> ] [ -h ] [ --help ]" 
    echo
    echo "Examples:"
    echo "                 $0 -d /dir/with/some/files -c 'gzip '" 
    echo "                 $0 -d /dir/with/some/files -c 'gzip \"\$ITEM\"' -D 5" 
    echo "                 $0 -d /dir/with/some/files -c 'cp \"\$ITEM\" /tmp' -p 2" 
}

showusage_normal () {

    echo 
    echo "|P|P|S|S| $SCRIPT_NAME $SCRIPT_VERSION"
    echo 
    echo "PPSS is a Bash shell script that executes commands in parallel on a set  "
    echo "of items, such as files in a directory, or lines in a file."
    echo 
    echo "This short summary only discusses options for stand-alone mode. for a "
    echo "complete listing of all options, run PPSS with the options --help"
    echo
    echo "Usage $0 [ options ]"
    echo
    echo -e "--command | -c     Command to execute. Syntax: '<command> ' including the single quotes."
    echo -e "                   Example: -c 'ls -alh '. It is also possible to specify where an item "
    echo -e "                   must be inserted: 'cp \"\$ITEM\" /somedir'."
    echo 
    echo -e "--sourcedir | -d   Directory that contains files that must be processed. Individual files" 
    echo -e "                   are fed as an argument to the command that has been specified with -c." 
    echo 
    echo -e "--sourcefile | -f  Each single line of the supplied file will be fed as an item to the"
    echo -e "                   command that has been specified with -c."
    echo 
    echo -e "--config | -C      If the mode is config, a config file with the specified name will be"
    echo -e "                   generated based on all the options specified. In the other modes". 
    echo -e "                   this option will result in PPSS reading the config file and start"
    echo -e "                   processing items based on the settings of this file."
    echo
    echo -e "--enable-ht | -j   Enable hyperthreading. Is disabled by default."
    echo 
    echo -e "--log | -l         Sets the name of the log file. The default is ppss-log.txt."
    echo
    echo -e "--processes | -p   Start the specified number of processes. Ignore the number of available"
    echo -e "                   CPUs."
    echo
    echo -e "--delay | -D       Adds an initial random delay to the start of all parallel jobs to spread"
    echo -e "                   the load. The delay is only used at the start of all 'threads'."
    echo 
    echo -e "Example: encoding some wav files to mp3 using lame:"
    echo 
    echo -e "$0 -d /path/to/wavfiles -c 'lame '" 
    echo 
    echo -e "Extended usage: use --help"
    echo
}

if [ "$#" == "0" ]
then
    showusage_short
    exit 1
fi

showusage_long () {
    
    echo 
    echo "|P|P|S|S| $SCRIPT_NAME $SCRIPT_VERSION"
    echo 
    echo "PPSS is a Bash shell script that executes commands in parallel on a set  "
    echo "of items, such as files in a directory, or lines in a file."
    echo 
    echo "Usage: $0 [ MODE ] [ options ]"
    echo 
    echo "Modes are optional and mainly used for running in distributed mode. Modes are:"
    echo 
    echo " config       Generate a config file based on the supplied option parameters."
    echo " deploy       Deploy PPSS and related files on the specified nodes."
    echo " erase        Erase PPSS and related files from the specified nodes."
    echo
    echo " start        Starting PPSS on nodes."
    echo " pause        Pausing PPSS on all nodes."
    echo " stop         Stopping PPSS on all nodes."
    echo " node         Running PPSS as a node, requires additional options."
    echo 
    echo "Options are:"
    echo 
    echo -e "--command | -c     Command to execute. Syntax: '<command> ' including the single quotes."
    echo -e "                   Example: -c 'ls -alh '. It is also possible to specify where an item "
    echo -e "                   must be inserted: 'cp \"\$ITEM\" /somedir'."
    echo 
    echo -e "--sourcedir | -d   Directory that contains files that must be processed. Individual files" 
    echo -e "                   are fed as an argument to the command that has been specified with -c." 
    echo 
    echo -e "--sourcefile | -f  Each single line of the supplied file will be fed as an item to the"
    echo -e "                   command that has been specified with -c."
    echo 
    echo -e "--config | -C      If the mode is config, a config file with the specified name will be"
    echo -e "                   generated based on all the options specified. In the other modes". 
    echo -e "                   this option will result in PPSS reading the config file and start"
    echo -e "                   processing items based on the settings of this file."
    echo
    echo -e "--enable-ht | -j   Enable hyperthreading. Is disabled by default."
    echo 
    echo -e "--log | -l         Sets the name of the log file. The default is ppss-log.txt."
    echo
    echo -e "--processes | -p   Start the specified number of processes. Ignore the number of available"
    echo -e "                   CPUs."
    echo
    echo -e "--delay | -D       Adds an initial random delay to the start of all parallel jobs to spread"
    echo -e "                   the load. The delay is only used at the start of all 'threads'."
    echo 
    echo -e "The following options are used for distributed execution of PPSS."
    echo 
    echo -e "--master | -m      Specifies the SSH server that is used for communication between nodes."
    echo -e "                   Using SSH, file locks are created, informing other nodes that an item "
    echo -e "                   is locked. Also, often items, such as files, reside on this host. SCP "
    echo -e "                   is used to transfer files from this host to nodes for local procesing."
    echo
    echo -e "--node | -n        File containig a list of nodes that act as PPSS clients. One IP / DNS "
    echo -e "                   name per line."
    echo
    echo -e "--key | -k         The SSH key that a node uses to connect to the master."
    echo
    echo -e "--known-hosts | -K The file that contains the server public key. Can often be found on  "
    echo -e "                   hosts that already once connected to the server. See the file "
    echo -e "                   ~/.ssh/known_hosts or else, manualy connect once and check this file."
    echo
    echo -e "--user | -u        The SSH user name that is used when logging in into the master SSH"
    echo -e "                   server."
    echo 
    echo -e "--script | -S      Specifies the script/program that must be copied to the nodes for "
    echo -e "                   execution through PPSS. Only used in the deploy mode."
    echo -e "                   This option should be specified if necessary when generating a config."
    echo
    echo -e "--transfer | -t    This option specifies that an item will be downloaded by the node "
    echo -e "                   from the server or share to the local node for processing."
    echo 
    echo -e "--no-scp | -b      Do not use scp for downloading items. Use cp instead. Assumes that a"
    echo -e "                   network file system (NFS/SMB) is mounted under a local mountpoint."
    echo 
    echo -e "--outputdir | -o   Directory on server where processed files are put. If the result of "
    echo -e "                   encoding a wav file is an mp3 file, the mp3 file is put in the "
    echo -e "                   directory specified with this option."
    echo 
    echo -e "--homedir | -H     Directory in which directory PPSS is installed on the node."
    echo -e "                   Default is 'ppss'."
    echo 
    echo -e "Example: encoding some wav files to mp3 using lame:"
    echo 
    echo -e "$0 -c 'lame ' -d /path/to/wavfiles -j " 
    echo 
    echo -e "Running PPSS based on a configuration file."
    echo
    echo -e "$0 -C config.cfg"
    echo 
    echo -e "Running PPSS on a client as part of a cluster."
    echo 
    echo -e "$0 -d /somedir -c 'cp "$ITEM" /some/destination' -s 10.0.0.50 -u ppss -t -k ppss-key.key"
    echo    
}

kill_process () {

    echo "$KILL_KEY" >> "$FIFO"
    }

exec_cmd () { 


    CMD="$1"

    if [ ! -z "$SSH_SERVER" ] 
    then
        ssh $SSH_OPTS $SSH_KEY $USER@$SSH_SERVER $CMD
        return $?
    else
        eval "$CMD"
        return $?
    fi
}

# this function makes remote or local checking of existence of items transparent.
does_file_exist () {

    FILE="$1"
    `exec_cmd "ls -1 $FILE" >> /dev/null 2>&1`
    if [ "$?" == "0" ]
    then
        return 0
    else 
        return 1
    fi
}

check_for_interrupt () {

    does_file_exist "$STOP_SIGNAL"
    if [ "$?" == "0" ]
    then
        set_status "STOPPED"
        log INFO "STOPPING job. Stop signal found."
        STOP="1"
        return 1
    fi

    does_file_exist "$PAUSE_SIGNAL"
    if [ "$?" == "0" ]
    then
        set_status "PAUZED"
        log INFO "PAUSE: sleeping for $PAUSE_DELAY SECONDS."
        sleep $PAUSE_DELAY
        check_for_interrupt
    else
        set_status "RUNNING"
    fi
}

cleanup () {

    #log DEBUG "$FUNCNAME - Cleaning up all temp files and processes."
    
    if [ -e "$FIFO" ]
    then 
        rm $FIFO 
    fi

    if [ -e "$ARRAY_POINTER_FILE" ] 
    then
        rm $ARRAY_POINTER_FILE
    fi

    if [ -e "$GLOBAL_LOCK" ] 
    then
        rm -rf $GLOBAL_LOCK
    fi

    if [ -e "$SSH_SOCKET" ]
    then
        rm -rf "$SSH_SOCKET"
    fi

}

add_var_to_config () {
    
    if [ "$MODE" == "config" ]
    then

        VAR="$1"
        VALUE="$2"

        echo -e "$VAR=$VALUE" >> $CONFIG
    fi
}

# Process any command-line options that are specified."
while [ $# -gt 0 ]
do
    case $1 in
        --config|-C )
                        CONFIG="$2"

                        if [ "$MODE" == "config" ]
                        then
                            if [ -e "$CONFIG" ]
                            then
                                echo "Do want to overwrite existing config file?"
                                read yn
                                if [ "$yn" == "y" ]
                                then
                                    rm "$CONFIG"
                                else
                                    echo "Aborting..."
                                    cleanup
                                    exit
                                fi 
                            fi
                        fi

                        if [ ! "$MODE" == "config" ]
                        then
                            source $CONFIG
                        fi

                        if [ ! -z "$SSH_KEY" ]
                        then
                            SSH_KEY="-i $SSH_KEY"
                        fi

                        if [ ! -e "./known_hosts" ]
                        then
                            if [ -e $SSH_KNOWN_HOSTS ]
                            then
                                cat $SSH_KNOWN_HOSTS > ./known_hosts
                            else
                                echo "File $SSH_KNOWN_HOSTS does not exist."
                                exit
                            fi
                        fi

                        shift 2
                        ;;
             --node|-n ) 
                        NODES_FILE="$2"
                        add_var_to_config NODES_FILE "$NODES_FILE"
                        shift 2
                        ;;

       --sourcefile|-f )
                        INPUT_FILE="$2"
                        add_var_to_config INPUT_FILE "$INPUT_FILE"
                        shift 2
                        ;;
        --sourcedir|-d ) 
                        SRC_DIR="$2"
                        add_var_to_config SRC_DIR "$SRC_DIR"
                        shift 2
                        ;; 
             --delay|-D)
                        MAX_DELAY="$2"
                        add_var_to_config MAX_DELAY "$MAX_DELAY"
                        shift 2
                        ;;
          --command|-c ) 
                        COMMAND=$2
                        if [ "$MODE" == "config" ]
                        then
                            COMMAND=\'$COMMAND\'
                            add_var_to_config COMMAND "$COMMAND"
                        fi
                        shift 2
                        ;;

                    -h )
                        showusage_normal
                        exit 1;;
                 --help)
                        showusage_long
                        exit 1;;
          --homedir|-H )
                        if [ ! -z "$2" ]
                        then
                            PPSS_HOME_DIR="$2"
                            add_var_to_config PPSS_DIR $PPSS_HOME_DIR
                            shift 2
                        fi
                        ;;
                        
       --disable-ht|-j )
                        HYPERTHREADING=no
                        add_var_to_config HYPERTHREADING $HYPERTHREADING
                        shift 1
                        ;;
              --log|-l )
                        LOGFILE="$2"
                        add_var_to_config LOGFILE "$LOGFILE"
                        shift 2
                        ;;
       --workingdir|-w ) 
                        WORKINGDIR="$2"
                        add_var_to_config WORKINGDIR "$WORKINGDIR"
                        shift 2
                        ;;
              --key|-k )
                        SSH_KEY="$2"
                        add_var_to_config SSH_KEY "$SSH_KEY"
                        if [ ! -z "$SSH_KEY" ]
                        then
                            SSH_KEY="-i $SSH_KEY"
                        fi
                        shift 2
                        ;;
    --known-hosts | -K ) 
                        SSH_KNOWN_HOSTS="$2"
                        add_var_to_config SSH_KNOWN_HOSTS "$SSH_KNOWN_HOSTS"
                        shift 2
                        ;;
                            
          --no-scp |-b )
                        SECURE_COPY=0
                        add_var_to_config SECURE_COPY "$SECURE_COPY"
                        shift 1
                        ;;
        --outputdir|-o )
                        REMOTE_OUTPUT_DIR="$2"
                        add_var_to_config REMOTE_OUTPUT_DIR "$REMOTE_OUTPUT_DIR"
                        shift 2
                        ;;
        --processes|-p )
                        TMP="$2"
                        if [ ! -z "$TMP" ]
                        then
                            MAX_NO_OF_RUNNING_JOBS="$TMP"
                            add_var_to_config MAX_NO_OF_RUNNING_JOBS "$MAX_NO_OF_RUNNING_JOBS" 
                            shift 2
                        fi
                        ;;
        --master|-m ) 
                        SSH_SERVER="$2"
                        add_var_to_config SSH_SERVER "$SSH_SERVER"
                        shift 2
                        ;;
        --script|-S )
                        SCRIPT="$2"
                        add_var_to_config SCRIPT "$SCRIPT"
                        shift 2
                        ;;
        --transfer|-t )
                        TRANSFER_TO_SLAVE="1"    
                        add_var_to_config TRANSFER_TO_SLAVE "$TRANSFER_TO_SLAVE"
                        shift 1
                        ;;
        --user|-u )
                        USER="$2"
                        add_var_to_config USER "$USER"
                        shift 2
                        ;;

        --version|-v )
                        echo ""
                        echo "$SCRIPT_NAME version $SCRIPT_VERSION"
                        echo ""
                        exit 0
                        ;;
        * )
                        showusage_normal
                        exit 1;;
    esac
done


display_header () {

    log info ""
    log INFO "========================================================="
    log INFO "                       |P|P|S|S|                         "
    log INFO "$SCRIPT_NAME version $SCRIPT_VERSION"
    log INFO "========================================================="
    log INFO "Hostname:\t\t$HOSTNAME"
    log INFO "---------------------------------------------------------"
}


# Init all vars
init_vars () {


    if [ "$ARCH" == "Darwin" ]
    then
        MIN_JOBS=4
    elif [ "$ARCH" == "Linux" ]
    then
        MIN_JOBS=3
    fi

    if [ -e "$LOGFILE" ]
    then
        rm $LOGFILE
    fi
    
    display_header

    if [ -z "$COMMAND" ]
    then
        echo
        log ERROR "No command specified."
        echo
        showusage_normal
        cleanup
        exit 1
    fi

    echo 0 > $ARRAY_POINTER_FILE

    FIFO=$PPSS_DIR/fifo-$RANDOM-$RANDOM

    if [ ! -e "$FIFO" ]
    then    
        mkfifo -m 600 $FIFO
    fi

    exec 42<> $FIFO

    set_status "RUNNING" 

    if [ -e "$CPUINFO" ]
    then
        CPU=`cat /proc/cpuinfo | grep 'model name' | cut -d ":" -f 2 | sed -e s/^\ //g | sort | uniq`
        log INFO "CPU: $CPU"
    elif [ "$ARCH" == "Darwin" ]
    then
        MODEL=`system_profiler SPHardwareDataType | grep "Processor Name" | cut -d ":" -f 2`
        SPEED=`system_profiler SPHardwareDataType | grep "Processor Speed" | cut -d ":" -f 2`
        log INFO "CPU: $MODEL $SPEED"
    elif [ "$ARCH" == "SunOS" ]
    then
        CPU=`psrinfo -v | grep MHz | cut -d " " -f 4,8 | awk '{ printf ("Processor architecture: %s @ %s MHz.\n", $1,$2) }' | head -n 1`
        
        log INFO "$CPU"
    else
        log INFO "CPU: Cannot determine. Provide a patch for your arch!"
        log INFO "Arch is $ARCH"
    fi 

    if [ -z "$MAX_NO_OF_RUNNING_JOBS" ]
    then 
        get_no_of_cpus $HYPERTHREADING
    fi

    does_file_exist "$JOB_LOG_DIR"
    if [ ! "$?" == "0" ]
    then
        log DEBUG "Job log directory $JOB_lOG_DIR does not exist. Creating."
        exec_cmd "mkdir -p $JOB_LOG_DIR"
    else
        log DEBUG "Job log directory $JOB_LOG_DIR exists."
    fi

    does_file_exist "$ITEM_LOCK_DIR"
    if [ ! "$?" == "0" ] && [ ! -z "$SSH_SERVER" ]
    then
        log DEBUG "Creating remote item lock dir."
        exec_cmd "mkdir $ITEM_LOCK_DIR"
    fi

    if [ ! -e "$JOB_LOG_DIR" ]
    then
        mkdir -p "$JOB_LOG_DIR"
    fi

    does_file_exist "$REMOTE_OUTPUT_DIR"
    if [ ! "$?" == "0" ]
    then
        log ERROR "Remote output dir $REMOTE_OUTPUT_DIR does not exist."
        set_status STOPPED
        cleanup
        exit
    fi

    if [ ! -e "$PPSS_LOCAL_TMPDIR" ]
    then
        mkdir "$PPSS_LOCAL_TMPDIR"
    fi

    if [ ! -e "$PPSS_LOCAL_OUTPUT" ] 
    then
        mkdir "$PPSS_LOCAL_OUTPUT"
    fi
}

get_status () {

    STATUS=`cat "$NODE_SATUS"`
    echo "$STATUS"
}

set_status () {

    STATUS="$1"
    echo "$HOSTNAME $STATUS" > "$NODE_STATUS"
}


expand_str () {

    STR=$1
    LENGTH=$TYPE_LENGTH
    SPACE=" "

    while [ "${#STR}" -lt "$LENGTH" ]
    do
        STR=$STR$SPACE
    done

    echo "$STR"
}

log () {
    
    # Type 'INFO' is logged to the screen
    # Any other log-type is only logged to the logfile.

    TYPE="$1"
    MESG="$2"
    TYPE_LENGTH=5 

    TYPE_EXP=`expand_str "$TYPE"`

    DATE=`date +%b\ %d\ %H:%M:%S`
    PREFIX="$DATE: ${TYPE_EXP:0:$TYPE_LENGTH}"
    PREFIX_SMALL="$DATE: "

    LOG_MSG="$PREFIX $MESG"
    ECHO_MSG="$PREFIX_SMALL $MESG"

    echo -e "$LOG_MSG" >> "$LOGFILE"

    if [ "$TYPE" == "INFO" ] || [ "$TYPE" == "ERROR" ] || [ "$TYPE" == "WARN" ]
    then
        echo -e "$ECHO_MSG"
    fi

}

check_status () {

    ERROR="$1"
    FUNCTION="$2"
    MESSAGE="$3"

    if [ ! "$ERROR" == "0" ]
    then
        log INFO "$FUNCTION - $MESSAGE"
        set_status STOPPED 
        cleanup
        exit 1
    fi

}

erase_ppss () {

    
    echo "Are you realy sure you want to erase PPSS from all nodes!? (YES/NO)"
    read YN

    if [ "$YN" == "yes" ] || [ "$YN" == "YES" ] 
    then
        for NODE in `cat $NODES_FILE`
        do
            log INFO "Erasing PPSS homedir $PPSS_DIR from node $NODE."
            ssh -q $SSH_KEY $SSH_OPTS $USER@$NODE "rm -rf $PPSS_HOME_DIR"
        done
    else
        log INFO "Aborting.."
    fi
    sleep 1
}

deploy () {

    NODE="$1"

    SSH_OPTS_NODE="-o BatchMode=yes -o ControlPath=socket-%h \
                           -o GlobalKnownHostsFile=./known_hosts \
                           -o ControlMaster=auto \
                           -o Cipher=blowfish \
                           -o ConnectTimeout=5 "

    ERROR=0
    set_error () {

        if [ ! "$1" == "0" ]
        then
            ERROR=1 
        fi
    }

    ssh -q -o ConnectTimeout=5 $SSH_KEY $USER@$NODE exit 0
    set_error "$?"
    if [ ! "$ERROR" == "0" ]
    then
        log ERROR "Cannot connect to node $NODE."
        return 
    fi

    ssh -N -M $SSH_OPTS_NODE $SSH_KEY $USER@$NODE &
    SSH_PID=$!

    KEY=`echo $SSH_KEY | cut -d " " -f 2` 

    sleep 1.1

    ssh -q $SSH_OPTS_NODE $SSH_KEY $USER@$NODE "cd ~ && mkdir $PPSS_HOME_DIR >> /dev/null 2>&1" 
    scp -q $SSH_OPTS_NODE $SSH_KEY $0 $USER@$NODE:~/$PPSS_HOME_DIR
    set_error $?
    scp -q $SSH_OPTS_NODE $SSH_KEY $KEY $USER@$NODE:~/$PPSS_HOME_DIR
    set_error $?
    scp -q $SSH_OPTS_NODE $SSH_KEY $CONFIG $USER@$NODE:~/$PPSS_HOME_DIR
    set_error $?
    scp -q $SSH_OPTS_NODE $SSH_KEY known_hosts $USER@$NODE:~/$PPSS_HOME_DIR
    set_error $?
    if [ ! -z "$SCRIPT" ]
    then
        scp -q $SSH_OPTS_NODE $SSH_KEY $SCRIPT $USER@$NODE:~/$PPSS_HOME_DIR
        set_error $?
    fi

    if [ ! -z "$INPUT_FILE" ]
    then
        scp -q $SSH_OPTS_NODE $SSH_KEY $INPUT_FILE $USER@$NODE:~/$PPSS_HOME_DIR
        set_error $?
    fi

    if [ "$ERROR" == "0" ]
    then
        log INFO "PPSS installed on node $NODE."
    else
        log INFO "PPSS failed to install on $NODE."
    fi

    kill $SSH_PID
}

deploy_ppss () {

    
    if [ -z "$NODES_FILE" ]
    then
        log INFO "ERROR - are you using the right option? -C ?"
        set_status ERROR
        cleanup 
        exit 1
    fi
    
    KEY=`echo $SSH_KEY | cut -d " " -f 2` 
    if [ -z "$KEY" ] || [ ! -e "$KEY" ]
    then
        log ERROR "Nodes require a key file."
        cleanup
        set_status "ERROR"
        exit 1
    fi

    if [ ! -e "$SCRIPT" ] && [ ! -z "$SCRIPT" ]
    then
        log ERROR "Script $SCRIPT not found."
        set_status "ERROR"
        cleanup
        exit 1
    fi

    INSTALLED_ON_SSH_SERVER=0
    if [ ! -e "$NODES_FILE" ]
    then
        log ERROR "File $NODES with list of nodes does not exist."
        set_status ERROR
        cleanup
        exit 1
    else
        for NODE in `cat $NODES_FILE` 
        do
            deploy "$NODE" &
            sleep 0.1
            if [ "$NODE" == "$SSH_SERVER" ]
            then   
                log DEBUG "SSH SERVER $SSH_SERVER is also a node."
                INSTALLED_ON_SSH_SERVER=1
                exec_cmd "mkdir -p $PPSS_HOME_DIR/$JOB_LOG_DIR"
                exec_cmd "mkdir -p $ITEM_LOCK_DIR"
            fi
        done
        if [ "$INSTALLED_ON_SSH_SERVER" == "0" ]
        then
            log DEBUG "SSH SERVER $SSH_SERVER is not a node."
            deploy "$SSH_SERVER"
            exec_cmd "mkdir -p $PPSS_HOME_DIR/$JOB_LOG_DIR"
            exec_cmd "mkdir -p $ITEM_LOCK_DIR"
        fi
    fi
}

start_ppss_on_node () {

    NODE="$1"

    log INFO "Starting PPSS on node $NODE."
    ssh $SSH_KEY $USER@$NODE "cd $PPSS_HOME_DIR ; screen -d -m -S PPSS ~/$PPSS_HOME_DIR/$0 --config ~/$PPSS_HOME_DIR/$CONFIG" 
}


test_server () {

    # Testing if the remote server works as expected.
    if [ ! -z "$SSH_SERVER" ] 
    then 
 
        exec_cmd "date >> /dev/null"
        check_status "$?" "$FUNCNAME" "Server $SSH_SERVER could not be reached"

        ssh -N -M $SSH_OPTS $SSH_KEY $USER@$SSH_SERVER &
        SSH_MASTER_PID="$!"
    else
        log DEBUG "No remote server specified, assuming stand-alone mode."
    fi
}

get_no_of_cpus () {

    # Use hyperthreading or not?
    HPT=$1
    NUMBER=""

    if [ -z "$HPT" ]
    then
        HPT=yes
    fi

    got_cpu_info () {

    ERROR="$1"
    check_status "$ERROR" "$FUNCNAME" "cannot determine number of cpu cores. Specify with -p." 

    }

    if [ "$HPT" == "yes" ]
    then
        if [ "$ARCH" == "Linux" ]
        then
            NUMBER=`grep ^processor $CPUINFO | wc -l`
            got_cpu_info "$?"

        elif [ "$ARCH" == "Darwin" ]
        then
            NUMBER=`sysctl -a hw | grep -w logicalcpu | awk '{ print $2 }'`
            got_cpu_info "$?"

        elif [ "$ARCH" == "FreeBSD" ]
        then
            NUMBER=`sysctl hw.ncpu | awk '{ print $2 }'`
            got_cpu_info "$?"

        elif [ "$ARCH" == "SunOS" ]
        then
            NUMBER=`psrinfo | grep on-line | wc -l`
            got_cpu_info "$?"
        else
            if [ -e "$CPUINFO" ]
            then
                NUMBER=`grep ^processor $CPUINFO | wc -l`
                got_cpu_info "$?"
            fi 
        fi

        if [ ! -z "$NUMBER" ]
        then
            log INFO "Found $NUMBER logic processors."
        fi

    elif [ "$HPT" == "no" ]
    then
        log INFO "Hyperthreading is disabled."

        if [ "$ARCH" == "Linux" ]
        then
            PHYSICAL=`grep 'physical id' $CPUINFO`
            if [ "$?" == "0" ]
            then
                PHYSICAL=`grep 'physical id' $CPUINFO | sort | uniq | wc -l`
                if [ "$PHYSICAL" == "1" ]
                then
                    log INFO "Found $PHYSICAL physical CPU."
                else
                    log INFO "Found $PHYSICAL physical CPUs."
                fi

                TMP=`grep 'core id' $CPUINFO` 
                if [ "$?" == "0" ]
                then
                    log DEBUG "Starting job only for each physical core on all physical CPU(s)."
                    NUMBER=`grep 'core id' $CPUINFO | sort | uniq | wc -l` 
                    log INFO "Found $NUMBER physical cores."
                else
                    log INFO "Single core processor(s) detected."
                    log INFO "Starting job for each physical CPU."
                    NUMBER=$PHYSICAL
                fi
            else
                log INFO "No 'physical id' section found in $CPUINFO, typical for older cpus."            
                NUMBER=`grep ^processor $CPUINFO | wc -l`
                got_cpu_info "$?"
            fi
        elif [ "$ARCH" == "Darwin" ]
        then
            NUMBER=`sysctl -a hw | grep -w physicalcpu | awk '{ print $2 }'`
            got_cpu_info "$?"
        elif [ "$ARCH" == "FreeBSD" ]
        then
            NUMBER=`sysctl hw.ncpu | awk '{ print $2 }'`
            got_cpu_info "$?"
        else
            NUMBER=`cat $CPUINFO | grep "cpu cores" | cut -d ":" -f 2 | uniq | sed -e s/\ //g`
            got_cpu_info "$?"
        fi

    fi

    if [ ! -z "$NUMBER" ] 
    then
        MAX_NO_OF_RUNNING_JOBS=$NUMBER
    else
        log ERROR "Number of CPUs not obtained."
        log ERROR "Please specify manually with -p."
        set_status "ERROR"
        exit 1
    fi
}

random_delay () {

    ARGS="$1"

    if [ -z "$ARGS" ]
    then
        log ERROR "$FUNCNAME Function random delay, no argument specified."
        set_status ERROR
        exit 1
    fi

    NUMBER=$RANDOM
    let "NUMBER %= $ARGS"
    sleep "$NUMBER"
}


global_lock () {

    mkdir $GLOBAL_LOCK > /dev/null 2>&1
    ERROR="$?"

    if [ ! "$ERROR" == "0" ]
    then
        return 1
    else
        return 0
    fi
}

get_global_lock () {

    while true
    do
        global_lock
        ERROR="$?"
        if [ ! "$ERROR" == "0" ]
        then
            random_delay $MAX_LOCK_DELAY
            continue
        else
            break
        fi
    done
}

release_global_lock () {

    rm -rf "$GLOBAL_LOCK"
}

are_jobs_running () {
   
    NUMBER_OF_PROCS=`jobs | wc -l`
    if [ "$NUMBER_OF_PROCS" -gt "1" ]
    then
        return 0
    else
        return 1
    fi
}

escape_item () {

    TMP="$1"

    ITEM_ESCAPED=`echo "$TMP" | \
            sed s/\\ /\\\\\\\\\\\\\\ /g | \
            sed s/\\'/\\\\\\\\\\\\\\'/g | \
            sed s/\\\`/\\\\\\\\\\\\\\\`/g | \
            sed s/\\|/\\\\\\\\\\\\\\|/g | \
            sed s/\&/\\\\\\\\\\\\\\&/g | \
            sed s/\;/\\\\\\\\\\\\\\;/g | \
            sed s/\(/\\\\\\\\\\(/g | \
            sed s/\)/\\\\\\\\\\)/g ` 
}

download_item () {

    ITEM="$1"
    ITEM_NO_PATH=`basename "$ITEM"`

    if [ "$TRANSFER_TO_SLAVE" == "1" ]
    then
        log DEBUG "Transfering item $ITEM_NO_PATH to local disk."
        if [ "$SECURE_COPY" == "1" ] && [ ! -z "$SSH_SERVER" ] 
        then
            if [ ! -z "$SRC_DIR" ]
            then
                ITEM_PATH="$SRC_DIR/$ITEM"
            else
                ITEM_PATH="$ITEM"
            fi 
            
            escape_item "$ITEM_PATH" 

            scp -q $SSH_OPTS $SSH_KEY $USER@$SSH_SERVER:"$ITEM_ESCAPED" ./$PPSS_LOCAL_TMPDIR
            log DEBUG "Exit code of remote transfer is $?"
        else
            cp "$ITEM" ./$PPSS_LOCAL_TMPDIR 
            log DEBUG "Exit code of local transfer is $?"
        fi
    else
        log DEBUG "No transfer of item $ITEM_NO_PATH to local workpath."
    fi
}

upload_item () {


    ITEM="$1"
    ITEMDIR="$2"

    if [ "$TRANSFER_TO_SLAVE" == "0" ]
    then
        log DEBUG "File transfer is disabled."
        return 0
    fi

    log DEBUG "Uploading item $ITEM."
    if [ "$SECURE_COPY" == "1" ]
    then
        escape_item "$REMOTE_OUTPUT_DIR$ITEMDIR"
        DIR_ESCAPED="$ITEM_ESCAPED"

        scp -q $SSH_OPTS $SSH_KEY "$ITEM"/* $USER@$SSH_SERVER:"$DIR_ESCAPED" 
        ERROR="$?"
        if [ ! "$ERROR" == "0" ]
        then
            log ERROR "Uploading of $ITEM via SCP failed."
        else
            log DEBUG "Upload of item $ITEM success" 
            rm -rf ./"$ITEM"
        fi
    else    
        cp "$ITEM" "$REMOTE_OUTPUT_DIR"
        ERROR="$?"
        if [ ! "$ERROR" == "0" ]
        then
            log DEBUG "ERROR - uploading of $ITEM vi CP failed."
        fi
    fi
}

lock_item () {
    
    if [ ! -z "$SSH_SERVER" ]
    then
        ITEM="$1"

        LOCK_FILE_NAME=`echo "$ITEM" | \
        sed s/^\\\.//g | \
        sed s/^\\\.\\\.//g | \
        sed s/^\\\///g | \
        sed s/\\\//\\\\\\ /g | \
        sed s/\\ /\\\\\\\\\\\\\\ /g | \
        sed s/\\'/\\\\\\\\\\\\\\'/g | \
        sed s/\&/\\\\\\\\\\\\\\&/g | \
        sed s/\;/\\\\\\\\\\\\\\;/g | \
        sed s/\(/\\\\\\\\\\(/g | \
        sed s/\)/\\\\\\\\\\)/g ` 

        ITEM_LOCK_FILE="$ITEM_LOCK_DIR/$LOCK_FILE_NAME"
        log DEBUG "Trying to lock item $ITEM - $ITEM_LOCK_FILE."
        exec_cmd "mkdir $ITEM_LOCK_FILE >> /dev/null 2>&1"
        ERROR="$?"

        if [ "$ERROR" == "$?" ]
        then
            exec_cmd "touch $ITEM_LOCK_FILE/$HOSTNAME"      # Record that item is claimed by node x.
        fi

        return "$ERROR"
    fi
}

get_all_items () {

    count=0

    if [ -z "$INPUT_FILE" ]
    then
        if [ ! -z "$SSH_SERVER" ] # Are we running stand-alone or as a slave?"
        then
            ITEMS=`exec_cmd "ls -1 $SRC_DIR"`
            check_status "$?" "$FUNCNAME" "Could not list files within remote source directory."
        else 
            if [ -e "$SRC_DIR" ]
            then
                ITEMS=`ls -1 $SRC_DIR`
            else
                ITEMS=""
            fi
        fi
        IFS=$'\n'

        for x in $ITEMS
        do
            ARRAY[$count]="$x"
            ((count++))
        done
        IFS=$IFS_BACKUP
    else
        if [ ! -z "$SSH_SERVER" ] # Are we running stand-alone or as a slave?"
        then
            log DEBUG "Running as slave, input file has been pushed (hopefully)."
            if [ ! -e "$INPUT_FILE" ]
            then
                log ERROR "Input file $INPUT_FILE does not exist."
                set_status "ERROR"
                cleanup 
                exit 1
            fi
        fi
    
        exec 10<"$INPUT_FILE"

        while read LINE <&10
        do
            ARRAY[$count]=$LINE
            ((count++))
        done
  
    fi
    exec 10>&-

    SIZE_OF_ARRAY="${#ARRAY[@]}"
    if [ "$SIZE_OF_ARRAY" -le "0" ]
    then
        log ERROR "Source file/dir seems to be empty."
        set_status STOPPED
        cleanup
        exit 1
    fi
}

get_item () {

    check_for_interrupt

    if [ "$STOP" == "1" ]
    then
        return 1
    fi
    
    get_global_lock

    SIZE_OF_ARRAY="${#ARRAY[@]}"

    # Return error if the array is empty.
    if [ "$SIZE_OF_ARRAY" -le "0" ]
    then
        release_global_lock
        return 1
    fi

    # This variable is used to walk thtough all array items.
    ARRAY_POINTER=`cat $ARRAY_POINTER_FILE`
    
    # Check if all items have been processed.
    if [ "$ARRAY_POINTER" -ge "$SIZE_OF_ARRAY" ]
    then
        release_global_lock
        echo -en "\033[1A"
        return 1
    fi

        # Select an item. 
    ITEM="${ARRAY[$ARRAY_POINTER]}" 
    if [ -z "$ITEM" ]
    then
        ((ARRAY_POINTER++))
        echo $ARRAY_POINTER > $ARRAY_POINTER_FILE
        release_global_lock
        get_item
    else
        ((ARRAY_POINTER++))
        echo $ARRAY_POINTER > $ARRAY_POINTER_FILE
        lock_item "$ITEM"
        if [ ! "$?" == "0" ]
        then
            log DEBUG "Item $ITEM is locked."
            release_global_lock
            get_item
        else
            log DEBUG "Got lock on $ITEM, processing."
            release_global_lock
            download_item "$ITEM"
            return 0
        fi
    fi
}

start_single_worker () {

    get_item
    ERROR=$?
    if [ ! "$ERROR" == "0" ]
    then
        # If no more items are available, the listener should be
        # informed that a worker just finished / died.
        # Tis allows the listener to determine if all processes
        # are finished and it is time to stop.
        echo
        echo "$STOP_KEY" > $FIFO
        return 1
    else
        get_global_lock
        echo "$ITEM" > $FIFO
        release_global_lock
        return 0
    fi
}


elapsed () {

    BEFORE="$1"
    AFTER="$2"

    ELAPSED="$(expr $AFTER - $BEFORE)"

    REMAINDER="$(expr $ELAPSED % 3600)"
    HOURS="$(expr $(expr $ELAPSED - $REMAINDER) / 3600)"

    SECS="$(expr $REMAINDER % 60)"
    MINS="$(expr $(expr $REMAINDER - $SECS) / 60)"

    echo "Elapsed time (h:m:s): $HOURS:$MINS:$SECS"
}

commando () {

    ITEM="$1"
    DIRNAME=`dirname "$ITEM"`
    ITEM_NO_PATH=`basename "$ITEM"`
    OUTPUT_DIR=$PPSS_LOCAL_OUTPUT/"$ITEM_NO_PATH"

    # This VAR can be used in scripts or command lines.
    OUTPUT_FILE="$ITEM_NO_PATH"

    log DEBUG "Processing item $ITEM"

    #Decide if an item must be transfered to the node.
    #or be processed in-place (NFS / SMB mount?)
    if [ "$TRANSFER_TO_SLAVE" == "0" ]
    then
        if [ -z "$SRC_DIR" ] && [ ! -z "$INPUT_FILE" ]
        then
            log DEBUG "Using item straight from INPUT FILE"
        else
            ITEM="$SRC_DIR/$ITEM"
        fi
    else
        ITEM="./$PPSS_LOCAL_TMPDIR/$ITEM_NO_PATH"
    fi

    LOG_FILE_NAME=`echo "$ITEM" | sed s/^\\\.//g | sed s/^\\\.\\\.//g | sed s/\\\///g`
    ITEM_LOG_FILE="$JOB_LOG_DIR/$LOG_FILE_NAME"

    mkdir -p "$OUTPUT_DIR"

    does_file_exist "$ITEM_LOG_FILE"
    if [ "$?" == "0" ]
    then
        log DEBUG "Skipping item $ITEM - already processed." 
    else
        
        ERROR=""

        # Some formatting of item log files. 
        DATE=`date +%b\ %d\ %H:%M:%S`
        echo "===== PPSS Item Log File =====" > "$ITEM_LOG_FILE"
        echo -e "Host:\t\t$HOSTNAME" >> "$ITEM_LOG_FILE"
        echo -e "Process:$PID" >> "$ITEM_LOG_FILE"
        echo -e "Item:\t\t$ITEM" >> "$ITEM_LOG_FILE"
        echo -e "Start date:\t$DATE" >> "$ITEM_LOG_FILE"
        echo -e "" >> "$ITEM_LOG_FILE"
        
        # The actual execution of the command.
        TMP=`echo $COMMAND | grep -i '$ITEM'`
        if [ "$?" == "0"  ]
        then 
            BEFORE="$(date +%s)"
            eval "$COMMAND" >> "$ITEM_LOG_FILE" 2>&1
            ERROR="$?"
            AFTER="$(date +%s)"
        else
            EXECME='$COMMAND"$ITEM" >> "$ITEM_LOG_FILE" 2>&1'
            BEFORE="$(date +%s)"
            eval "$EXECME"
            ERROR="$?"
            AFTER="$(date +%s)"
        fi

        echo -e "" >> "$ITEM_LOG_FILE"

        # Some error logging. Success or fail.
        if [ ! "$ERROR" == "0" ] 
        then
           echo -e "Status:\t\tFAILURE" >> "$ITEM_LOG_FILE"
        else
           echo -e "Status:\t\tSUCCESS" >> "$ITEM_LOG_FILE"
        fi

        #Remove the item after it has been processed as not to fill up disk space.
        if [ "$TRANSFER_TO_SLAVE" == "1" ]      
        then
            if [ -e "$ITEM" ]
            then
                rm "$ITEM"
            else        
                log DEBUG "ERROR Something went wrong removing item $ITEM from local work dir."
            fi

        fi

        NEWDIR="$REMOTE_OUTPUT_DIR/$DIRNAME"
        escape_item "$NEWDIR"
        DIR_ESCAPED="$ITEM_ESCAPED"

        exec_cmd "mkdir -p $DIR_ESCAPED"
        if [ "$DIRNAME" == "." ]
        then
            DIRNAME=""
        fi
        upload_item "$PPSS_LOCAL_OUTPUT/$ITEM_NO_PATH" "$DIRNAME"
        
        elapsed "$BEFORE" "$AFTER" >> "$ITEM_LOG_FILE"
        echo -e "" >> "$ITEM_LOG_FILE"

        if [ ! -z "$SSH_SERVER" ]
        then
            log DEBUG "Uploading item log file $ITEM_LOG_FILE to master ~/$PPSS_HOME_DIR/$JOB_LOG_DIR"
            scp -q $SSH_OPTS $SSH_KEY "$ITEM_LOG_FILE" $USER@$SSH_SERVER:~/$PPSS_HOME_DIR/$JOB_LOG_DIR
            if [ ! "$?" == "0" ]
            then
                log DEBUG "Uploading of item log file failed."
            fi
        fi
    fi

    start_single_worker
    return $?
}

# This is the listener service. It listens on the pipe for events.
# A job is executed for every event received.
# This listener enables fully asynchronous processing.
listen_for_job () {
    FINISHED=0
    DIED=0
    PIDS=""
    log DEBUG "Listener started."
    while read event <& 42
    do
        # The start_single_worker method sends a special signal to 
        # inform the listener that a worker is finished.
        # If all workers are finished, it is time to stop.
        # This mechanism makes PPSS asynchronous.

        # Gives a status update on the current progress..
        
        if [ "$event" == "$STOP_KEY"  ]
        then
            ((DIED++))
            if [ "$DIED" -ge "$MAX_NO_OF_RUNNING_JOBS" ] 
            then
                kill_process
            else
                RES=$((MAX_NO_OF_RUNNING_JOBS-DIED))
                if [ "$RES" == "1" ]
                then
                    log INFO "$((MAX_NO_OF_RUNNING_JOBS-DIED)) job is remaining.       "
                else
                    log INFO "$((MAX_NO_OF_RUNNING_JOBS-DIED)) jobs are remaining."
                    echo -en "\033[1A"
                fi
            fi
        elif [ "$event" == "$KILL_KEY" ]
        then
            for x in $PIDS
            do
                kill $x >> /dev/null 2>&1
            done
            if [ ! -z "$SSH_MASTER_PID" ]
            then
                kill "$SSH_MASTER_PID" 
            fi
            log INFO "Finished. Consult ./$JOB_LOG_DIR for job output."
            break
        else
            commando "$event" &
            PIDS="$PIDS $!"
            disown
        fi

        SIZE_OF_ARRAY="${#ARRAY[@]}"
        ARRAY_POINTER=`cat $ARRAY_POINTER_FILE`
        PERCENT=$((100 * $ARRAY_POINTER / $SIZE_OF_ARRAY ))
        if [ "$DIED" == "0" ] && [ "$FINISHED" == "0" ]
        then
            log INFO "Currently $PERCENT percent complete. Processed $ARRAY_POINTER of $SIZE_OF_ARRAY items." 
            if [ "$PERCENT" == "100" ]
            then
                FINISHED=1
            else
                echo -en "\033[1A"
            fi
        fi
    done

    set_status STOPPED
    log DEBUG "Listener stopped."
    cleanup
    exit
}

# This starts an number of parallel workers based on the # of parallel jobs allowed.
start_all_workers () {

    if [ "$MAX_NO_OF_RUNNING_JOBS" == "1" ]
    then
        log INFO "Starting $MAX_NO_OF_RUNNING_JOBS single worker."
    else
        log INFO "Starting $MAX_NO_OF_RUNNING_JOBS parallel workers."
    fi
    log INFO "---------------------------------------------------------"

    i=0
    while [ "$i" -lt "$MAX_NO_OF_RUNNING_JOBS" ]
    do
        start_single_worker
        ((i++))

        if [ ! "$MAX_DELAY" == "0" ]
        then
            random_delay "$MAX_DELAY"
        fi
    done
}

get_status_of_node () {

    NODE="$1"
    STATUS=`ssh -o ConnectTimeout=10 $SSH_KEY $USER@$NODE cat "$PPSS_HOME_DIR/$NODE_STATUS" 2>/dev/null`
    ERROR="$?"
    if [ ! "$ERROR" == "0" ]
    then
        STATUS="UNKNOWN"
    fi
    echo "$STATUS"
}

show_status () {

    source $CONFIG
    if [ ! -z "$SSH_KEY" ]
    then
        SSH_KEY="-i $SSH_KEY"
    fi

    if [ -z "$INPUT_FILE" ]
    then
        ITEMS=`exec_cmd "ls -1 $SRC_DIR | wc -l"`  
    else
        ITEMS=`exec_cmd "cat $PPSS_DIR/$INPUT_FILE | wc -l"`
    fi
    
    PROCESSED=`exec_cmd "ls -1 $ITEM_LOCK_DIR | wc -l"` 2>&1 >> /dev/null
    TMP_STATUS=$((100 * $PROCESSED / $ITEMS))

    log INFO "Status:\t\t$TMP_STATUS percent complete."

    if [ ! -z $NODES_FILE ]
    then
        TMP_NO=`cat $NODES_FILE | wc -l`
        log INFO "Nodes:\t $TMP_NO"
    fi
    log INFO "Items:\t\t$ITEMS"


    log INFO "---------------------------------------------------------"
    HEADER=`echo IP-address Hostname Processed Status | awk '{ printf ("%-16s %-18s % 10s %10s\n",$1,$2,$3,$4) }'`  
    log INFO "$HEADER"
    log INFO "---------------------------------------------------------"
    PROCESSED=0
    for x in `cat $NODES_FILE`
    do
        NODE=`get_status_of_node "$x" | awk '{ print $1 }'`
        if [ ! "$NODE" == "UNKNOWN" ]
        then
            STATUS=`get_status_of_node "$x" | awk '{ print $2 }'`
            RES=`exec_cmd "grep -i $NODE ~/$PPSS_HOME_DIR/$JOB_LOG_DIR/* 2>/dev/null | wc -l "`
            if [ ! "$?" == "0" ]
            then
                RES=0
            fi
        else
            STATUS="UNKNOWN"
            RES=0
        fi
        let PROCESSED=$PROCESSED+$RES
        LINE=`echo "$x $NODE $RES $STATUS" | awk '{ printf ("%-16s %-18s % 10s %10s\n",$1,$2,$3,$4) }'`
        log INFO "$LINE"
    done
    log INFO "---------------------------------------------------------"
    LINE=`echo $PROCESSED | awk '{ printf ("Total processed: % 29s\n",$1) }'`
    log INFO "$LINE"
}


# If this is called, the whole framework will execute.
main () {
    
    case $MODE in
              node )
                    init_vars
                    test_server
                    get_all_items
                    listen_for_job "$MAX_NO_OF_RUNNING_JOBS" & 2>&1 >> /dev/null
                    LISTENER_PID=$!
                    start_all_workers
                    ;;
             start )
                    # This option only starts all nodes.
                    display_header
                    if [ ! -e "$NODES_FILE" ]
                    then
                        log ERROR "File $NODES with list of nodes does not exist."
                        set_status STOPPED
                        cleanup
                        exit 1
                    else
                        for NODE in `cat $NODES_FILE`
                        do
                            start_ppss_on_node "$NODE"
                        done
                    fi
                    cleanup
                    exit 0
                    ;;
        config )
                    display_header
                    log INFO "Generating configuration file $CONFIG"
                    add_var_to_config PPSS_LOCAL_TMPDIR "$PPSS_LOCAL_TMPDIR"
                    add_var_to_config PPSS_LOCAL_OUTPUT "$PPSS_LOCAL_OUTPUT"
                    cleanup
                    exit 0
                    ;;

        stop )
                    display_header
                    log INFO "Stopping PPSS on all nodes."
                    exec_cmd "touch $STOP_SIGNAL"
                    cleanup
                    exit
                    ;;
        pause )
                    display_header
                    log INFO "Pausing PPSS on all nodes."
                    exec_cmd "touch $PAUSE_SIGNAL"
                    cleanup
                    exit
                    ;;
        continue )
                    display_header
                    if does_file_exist "$STOP_SIGNAL"
                    then
                        log INFO "Continuing processing, please use $0 start to start PPSS on al nodes."
                        exec_cmd "rm -f $STOP_SIGNAL"
                    fi
                    if does_file_exist "$PAUSE_SIGNAL"
                    then
                        log INFO "Continuing PPSS on all nodes."
                        exec_cmd "rm -f $PAUSE_SIGNAL"
                    fi
                    cleanup
                    exit 0
                    ;;
        deploy )
                    display_header
                    log INFO "Deploying PPSS on nodes."
                    deploy_ppss
                    wait
                    cleanup
                    exit 0
                    ;;
        status )
                    display_header
                    show_status
                    cleanup
                    exit 0
                    ;;
        erase )
                    display_header
                    log INFO "Erasing PPSS from all nodes."
                    erase_ppss
                    cleanup
                    exit 0
                    ;;
        kill )
                    for x in `ps ux | grep ppss | grep -v grep | grep bash | awk '{ print $2 }'`
                    do          
                         kill "$x"
                    done
                    cleanup
                    exit 0
                    ;;

        * )
                    init_vars
                    get_all_items
                    listen_for_job "$MAX_NO_OF_RUNNING_JOBS" & 2>&1 >> /dev/null
                    LISTENER_PID=$!
                    start_all_workers
                    ;;

    esac

}
# This command starts the that sets the whole framework in motion.
main

# Exit after all processes have finished.
wait
