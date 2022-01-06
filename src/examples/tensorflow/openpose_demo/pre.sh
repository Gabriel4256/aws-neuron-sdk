#!/bin/bash

wget -c --tries=2 $( wget -q -O - http://www.mediafire.com/file/qlzzr20mpocnpa3/graph_opt.pb | grep -o 'http*://download[^"]*' | tail -n 1 ) -O graph_opt.pb