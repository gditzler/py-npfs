# Neyman-Pearson Based Feature Selection (NPFS) post-hoc test

Python implementation of NPFS. It is still under development and is not considered stable. Contact <gregory.ditzler@gmail.com>.

There are types of feature subset selection problems that require that the size of the subset be specied prior to running the selection algorithm. NPFS works with the decisions of a base subset selection algorithm to determine an appropriate number of features to select given an initial starting point. NPFS uses the FEAST feature selection toolbox; however, the approach is not limited to using the this toolbox. 

# Module Installation 

```bash
  cd src/
  python setup.py build
  sudo python setup.py install 
``` 

# Citing NPFS

* Gregory Ditzler, Robi Polikar, Gail Rosen, "A Bootstrap Based Neyman–Pearson Test for Identifying Variable Importance," 2014, In press. ([link](http://gregoryditzler.files.wordpress.com/2014/05/tnnls2014.pdf))

# Word Cloud

![alt tag](https://raw.github.com/gditzler/NPFS/master/img/npfs.jpg)

