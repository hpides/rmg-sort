#!/bin/bash

clang-format -i src/*
yapf3 -i --recursive scripts/*.py
