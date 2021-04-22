@echo off
SetLocal


FOR %%n in (100 150 250) do (
FOR %%s in (quickshift felzenszwalb) do (
FOR %%r in (linear lasso ridge) do (
FOR %%x in (3) do (
FOR %%w in (true false) do (

echo python run.py -n %%n --seg %%s --reg %%r --sel %%x --weights %%w
call python run.py -n %%n --seg %%s --reg %%r --sel %%x --weights %%w

)
)
)
)
)