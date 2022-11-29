This is a collection of *necessary* tests 
to verify some of the basic functionality 
of the tools in calcom. 

It is not necessarily comprehensive.
It is not on the level of unit testing. 

To include your own test, make a new python file 
(it doesn't have to follow the existing naming 
convention - any .py file will be found) 
and implement a test() function that obeys 
the following conventions:

- runs with no arguments, 
- returns True if the functionality behaves as expected, 
- either returns False or raises an Exception if 
  something goes wrong.
- includes a docstring inside the beginning of the 
  test() function (triple tick marks) briefly 
  describing what you're checking with the test.

See test_000.py through test_004.py for some 
simple examples to get an idea; later 
examples test more complicated functionality.

======

To run the tests for yourself, either 
run sanity.py, or import sanity and then 
execute sanity.test(). Currently you need 
to be in the sanity/ folder, or imports 
will likely fail. This may or may not be 
fixed in the future.

-Nuch

README last updated 26 Feb 2019
