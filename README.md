# Crossword evolutionary generating algorythm
Crossword generator based on Evolutionary algorithm

## 🚀 How to run
```shell
git clone https://github.com/anijackich/Evolutionary-Crossword.git
cd Evolutionary-Crossword
python crossword.py
```
The program can process multiple input files with words for a crossword. Input files must be located in the `inputs` directory and have a name in format `inputM.txt`. Where `M` - the number of the input test.

After solving the test, the program prints the generated crossword and saves the output file with the solution storing the coordinates and directions (0 - horizontal, 1 - vertical) of the words. 

### Example
```
forming
realistic
adequate
origin
appearing
lord
iceland
legal
```
```
    0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19
 0                                                            
 1                                                            
 2                                                            
 3                                                            
 4                                                            
 5                                                            
 6                                                            
 7     a                                                      
 8     d                                                      
 9  l  e  g  a  l                                             
10     q     p     i              f                           
11     u     p     c              o                           
12     a     e     e              r                           
13     t     a     l              m                           
14     e     r  e  a  l  i  s  t  i  c                        
15           i     n              n                           
16  l        n     d              g                           
17  o  r  i  g  i  n                                          
18  r                                                         
19  d                                                          
```

## 📚 Words generator
You can use a script `words_generator.py` to generate random tests:
```shell
python words_generator.py
```

## 📊 Performance statistics
![](https://0x0.st/H3Kb.png)