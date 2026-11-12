#!/bin/bash
until grep -q DONE /Users/lucaszhang/squares-in-squares/poc/public_resweep_other/data/ver1.log; do sleep 10; done
