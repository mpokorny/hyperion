#!/bin/bash
set -ev

LOCKFILE=${CASACORE_DATA_DIR}/.lock
dotlockfile -l -r -1 ${LOCKFILE}
if [ ! -f ${CASACORE_DATA_DIR}/WSRT_Measures.ztar ]; then
    cd ${CASACORE_DATA_DIR}
    wget -t 0 -w 10 ftp://ftp.astron.nl/outgoing/Measures/WSRT_Measures.ztar
    tar zxf WSRT_Measures.ztar
fi
dotlockfile -u ${LOCKFILE}
