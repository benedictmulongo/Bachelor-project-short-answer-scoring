"""
Benedith Mulongo
The code is from the following website :

https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Longest_common_subsequence

It has not been implemented by myself partly because it is already implemented much better 
by smarter people and also because I want to use LCS in another application, it would have
take more time to do both.

 
"""

def LCS(X, Y):
    m = len(X)
    n = len(Y)
    C = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if X[i-1] == Y[j-1]: 
                C[i][j] = C[i-1][j-1] + 1
            else:
                C[i][j] = max(C[i][j-1], C[i-1][j])
    return C
    
def backTrack(C, X, Y, i, j):
    if i == 0 or j == 0:
        return ""
    elif X[i-1] == Y[j-1]:
        return backTrack(C, X, Y, i-1, j-1) + X[i-1]
    else:
        if C[i][j-1] > C[i-1][j]:
            return backTrack(C, X, Y, i, j-1)
        else:
            return backTrack(C, X, Y, i-1, j)
            
def backTrackAll(C, X, Y, i, j):
    if i == 0 or j == 0:
        return set([""])
    elif X[i-1] == Y[j-1]:
        return set([Z + X[i-1] for Z in backTrackAll(C, X, Y, i-1, j-1)])
    else:
        R = set()
        if C[i][j-1] >= C[i-1][j]:
            R.update(backTrackAll(C, X, Y, i, j-1))
        if C[i-1][j] >= C[i][j-1]:
            R.update(backTrackAll(C, X, Y, i-1, j))
        return R
        
def printDiff(C, X, Y, i, j):
    if i > 0 and j > 0 and X[i-1] == Y[j-1]:
        printDiff(C, X, Y, i-1, j-1)
        print("  " + X[i-1])
    else:
        if j > 0 and (i == 0 or C[i][j-1] >= C[i-1][j]):
            printDiff(C, X, Y, i, j-1)
            print("+ " + Y[j-1])
        elif i > 0 and (j == 0 or C[i][j-1] < C[i-1][j]):
            printDiff(C, X, Y, i-1, j)
            print("- " + X[i-1])
            
def compute_LCS(string_a, string_b, all = False):
    
    X = string_a
    Y = string_b
    m = len(X)
    n = len(Y)
    C = LCS(X, Y)
    
    back = backTrack(C, X, Y, m, n)
    if all:
        back = backTrackAll(C, X, Y, m, n)
        
    return back
    
    
def testDiff():
    X = [
        "This part of the document has stayed",
        "the same from version to version.",
        "",
        "This paragraph contains text that is",
        "outdated - it will be deprecated '''and'''",
        "deleted '''in''' the near future.",
        "",
        "It is important to spell check this",
        "dokument. On the other hand, a misspelled",
        "word isn't the end of the world.",
    ]
    Y = [
        "This is an important notice! It should",
        "therefore be located at the beginning of",
        "this document!",
        "",
        "This part of the document has stayed",
        "the same from version to version.",
        "",
        "It is important to spell check this",
        "document. On the other hand, a misspelled",
        "word isn't the end of the world. This",
        "paragraph contains important new",
        "additions to this document.",
    ]
    
    C = LCS(X, Y)
    printDiff(C, X, Y, len(X), len(Y))
    
def testLCS():
    
    X = "albastru"
    Y = "alabaster"
    m = len(X)
    n = len(Y)
    C = LCS(X, Y)
    
    print("Some LCS: ", backTrack(C, X, Y, m, n))
    print("All LCSs: ", backTrackAll(C, X, Y, m, n))

