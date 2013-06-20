mothur-test
===========

The mothur-test make file will work if mothur-test, mothur and gtest are under one directory.  Download and unzip gtest-1.6.0.zip from the googletest website http://code.google.com/p/googletest/.

parent/
       gtest-1.6.0/
       mothur/
       mothur-test/

Build the mothur object files with make or an IDE.  Change to the mothur-test directory and run make.  Then run ./svm_test.
