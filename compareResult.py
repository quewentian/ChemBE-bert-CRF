"""
group the token and the content of the test.txt, compare the predicted results and the original tags
"""

def writeToOneFile(data_dir,output_dir):
    testResult=''
    predictPath=output_dir+'/predict_file.txt'
    tokenPath=output_dir+'/token_test.txt'
    with open(predictPath,'r',encoding='UTF-8') as f:
        testResult=f.read()
    with open(tokenPath,'r',encoding='UTF-8') as f1:
        token=f1.read()

    """
    one passage is a list
    """
    testResult=testResult.split()#label's list
    token=token.split()#token's list

    """
    group the passage into sentences[[...one sentence],[one sentence]...[...]]
    """

    result=[]
    oneresult=[]
    tokenResult=[]
    oneToken=[]
    if len(testResult)==len(token):
        for i in range(len(token)):
            if token[i]=='[SEP]':
                result.append(oneresult)
                oneresult=[]
                tokenResult.append(oneToken)
                oneToken=[]
            if  token[i]!='[CLS]' and token[i]!='[SEP]':
                oneresult.append(testResult[i])
                oneToken.append(token[i])
    else:
        print('something wrong with the data!!! Please check it!')
        for i in range(len(testResult)):
            if token[i]=='[SEP]':
                result.append(oneresult)
                oneresult=[]
                tokenResult.append(oneToken)
                oneToken=[]
            if  token[i]!='[CLS]' and token[i]!='[SEP]':
                oneresult.append(testResult[i])
                oneToken.append(token[i])
    # print(result)
    # print(tokenResult)

    """
    read the test.txt file
    """
    content=[]
    testPath=data_dir+'/test.txt'
    with open(testPath,'r',encoding='UTF-8') as f1:
        line=f1.readline()
        while line:
            if not line.__contains__('-DOCSTART-') or line!='\n':
                content.append(line)
            line=f1.readline()

    """
    replace other symbols
    """
    clearContent=[]
    for j in range(len(content)):
        # print(content[j])
        if not content[j].__contains__('-DOCSTART-') and content[j]!='\n' and content[j]!='':
            # print(content[j].replace('\n',''),'   ##')
            clearContent.append(content[j].strip().replace('\n','').replace('\ufeff',''))
    # print(clearContent)

    """
    show the test.txt content with the form[[one sentence],[],[],...]
    """
    text=[]#####every sentence saved as a list[[],[],...]
    real=[]##########tags of every sentence saves as a list[[],[],...]
    onetext=[]#one sentence
    onereal=[]# tags of one sentence
    counter=0
    for w in range(len(clearContent)):
        counter+=1
        line1=clearContent[w].split()
        if line1[0]=='.':
            onetext.append(line1[0])
            text.append(onetext)
            onetext=[]
            # print(onereal)
            try:
                onereal.append(line1[1])
                real.append(onereal)
                onereal=[]
            except:
                print(line1,'   hhh',  counter)
        else:
            onetext.append(line1[0])
            try:
                onereal.append(line1[1])
            except:
                print(line1,'  error')


    """
    match the token and the original test content
    """
    deleteToken=[]
    deleteOneToken=[]
    deleteLabel=[]
    deleteOneLabel=[]
    deleteText=[]
    deleteOneText=[]

    for n in range(len(tokenResult)):
        counter2=0
        for b in range(len(tokenResult[n])):

            if counter2<len(text[n]):

                if b<len(tokenResult[n])-1:
                    print(text[n][counter2].lower().startswith(tokenResult[n][b]),'  YY')
                    if text[n][counter2].lower().startswith(tokenResult[n][b]):
                        # print(tokenResult[n][b],' 33')
                        deleteOneText.append(text[n][counter2])
                        counter2+=1
                        deleteOneLabel.append(result[n][b])

                if b == len(tokenResult[n]) - 1:
                    deleteOneLabel.append('<O>')
                    deleteOneText.append(text[n][counter2])
        deleteLabel.append(deleteOneLabel)
        deleteText.append(deleteOneText)
        deleteOneLabel=[]
        deleteOneText=[]
    # print(deleteLabel)
    # print(deleteText)

    outContent=''
    for h in range(len(deleteText)):
        for k in range(len(deleteText[h])):
            print(deleteText[h][k],'   ',deleteLabel[h][k])
            outContent+=deleteText[h][k]
            outContent +=" "
            outContent +=deleteLabel[h][k]
            outContent+='\n'
    outcomePath=output_dir+'/predictOutcome.txt'
    with open(outcomePath,'w',encoding='UTF-8') as f:
        f.write(outContent)
