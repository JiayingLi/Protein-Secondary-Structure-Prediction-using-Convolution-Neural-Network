def main():
    f = open('casp9.data','r')  
    result = list()  
    num_train = 300
    num_test = 30
    result1 = list()
    for line in open('casp9.data'):
        line = f.readline()  
        #print line
        result.append(line)
        num_train -= 1
        if num_train == 0:
            break
    for line in open('casp9.data'):
        
        line = f.readline()  
        #print line
        result1.append(line)
        num_test -= 1
        if num_test == 0:
            break
    #print result  
    f.close()                  
    open('train.data', 'w').write('%s' % ''.join(result))
    open('test.data', 'w').write('%s' % ''.join(result1))  
if __name__ == '__main__':
    main()