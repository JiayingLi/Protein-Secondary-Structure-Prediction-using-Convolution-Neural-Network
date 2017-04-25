
def main():
    f = open('valid.pssm','r')  
    result = list()  
    num_train = 1000
    num_test = 100
    result1 = list()
    num_proteins = int(f.readline())
    result.append(str(num_train)+'\n')
    result1.append(str(num_test)+'\n')
    print('Spliting' , num_train,'training and', num_test ,'testing proteins from ', num_proteins, 'samples')
    for n in range(num_train):
        m = int(f.readline())
        result.append(str(m)+'\n')
        for i in range(m):
            line = f.readline()
            result.append(line)
        line = f.readline()
        result.append(line)
    for n in range(num_test):
        m = int(f.readline())
        result1.append(str(m)+'\n')
        for i in range(m):
            line = f.readline()
            result1.append(line)
        line = f.readline()
        result1.append(line)
    #print result  
    f.close()                  
    open('train1000.pssm', 'w').write('%s' % ''.join(result))
    open('testing100.pssm', 'w').write('%s' % ''.join(result1))  
if __name__ == '__main__':
    main()