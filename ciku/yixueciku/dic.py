import struct
import os




#首先读入各个文件，将各个文件中的词都初始化为统一格式

def get_data_dic(file_path = './body中文身体部位名称.txt'):
    result = {}
    # 拼音表偏移，
    startPy = 0x1540;

    # 汉语词组表偏移
    startChinese = 0x2628;

    # 全局拼音表
    GPy_Table = {}

    # 原始字节码转为字符串
    def byte2str(data):
        pos = 0
        str = ''
        while pos < len(data):
            c = chr(struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0])
            if c != chr(0):
                str += c
            pos += 2
        return str

    # 获取拼音表
    def getPyTable(data):
        data = data[4:]
        pos = 0
        while pos < len(data):
            index = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
            pos += 2
            lenPy = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
            pos += 2
            py = byte2str(data[pos:pos + lenPy])
            GPy_Table[index] = py
            pos += lenPy

    # 获取一个词组的拼音
    def getWordPy(data):
        pos = 0
        ret = ''
        while pos < len(data):
            index = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
            ret += GPy_Table[index]
            pos += 2
        return ret

    # 读取中文表
    def getChinese(data):
        pos = 0
        while pos < len(data):
            # 同音词数量
            same = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
            # 拼音索引表长度
            pos += 2
            py_table_len = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
            # 拼音索引表
            pos += 2
            py = getWordPy(data[pos: pos + py_table_len])
            # 中文词组
            pos += py_table_len
            for i in range(same):
                # 中文词组长度
                c_len = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
                # 中文词组
                pos += 2
                word = byte2str(data[pos: pos + c_len])
                # 扩展数据长度
                pos += c_len
                ext_len = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
                # 词频
                pos += 2
                count = struct.unpack('H', bytes([data[pos], data[pos + 1]]))[0]
                # 保存
                GTable.append((count, py, word))
                # 到下个词的偏移位置
                pos += ext_len

    def scel2txt(file_name):
        # 读取文件
        with open(file_name, 'rb') as f:
            data = f.read()
        getPyTable(data[startPy:startChinese])
        getChinese(data[startChinese:])

    # 解析结果
    # 元组(词频,拼音,中文词组)的列表
    GTable = []


    if '中文身体部位名称.txt' in file_path:#处理的是body中文身体部位名称这个文件
        f = open(file_path, 'r', encoding = 'utf-8')
        ff = f.readlines()
        for line in ff:
            word = ""
            for i in range(len(line)):
                if line[i] == " ":
                    break
                word += line[i]
            if result.get(word, 0) == 0:
                result[word] = 1
            else:
                result[word] += 1
    if 'symptom.txt' in file_path:#处理的是body中文身体部位名称这个文件
        f = open(file_path, 'r', encoding = 'utf-8')
        ff = f.readlines()
        for line in ff:
            word = ""
            for i in range(len(line)):
                if line[i] == ",":
                    word = line[i+1:-1]
                    break
            if result.get(word, 0) == 0:
                result[word] = 1
            else:
                result[word] += 1
    if 'THUOCL_medical.txt' in file_path:#处理的是body中文身体部位名称这个文件
        f = open(file_path, 'r', encoding = 'utf-8')
        ff = f.readlines()
        for line in ff:
            word = ""
            for i in range(len(line)):
                if line[i] == "\t":
                    break
                word += line[i]
            if result.get(word, 0) == 0:
                result[word] = 1
            else:
                result[word] += 1
    if  'disease_new' in file_path:#处理的是disease_new这个文件
        f = open(file_path, 'r', encoding='utf-8')
        ff = f.readlines()
        for line in ff:
            word = line[:-1]
            if result.get(word, 0) == 0:
                result[word] = 1
            else:
                result[word] += 1
    if 'ICD10诊断.scel' in file_path:#处理的是ICD10诊断.scel，需要先转换为.txt文件
        scel2txt(file_path)#先将scel文件转换为txt文件a
        for count, py, word in GTable:
            if result.get(word, 0) == 0:
                result[word] = 1
            else:
                result[word] += 1

    if '医院电子病历词库.scel' in file_path:#处理的是ICD10诊断.scel，需要先转换为.txt文件
        scel2txt(file_path)#先将scel文件转换为txt文件a
        for count, py, word in GTable:
            if result.get(word, 0) == 0:
                result[word] = 1
            else:
                result[word] += 1

    if '症状.scel' in file_path:#处理的是ICD10诊断.scel，需要先转换为.txt文件
        scel2txt(file_path)#先将scel文件转换为txt文件a
        for count, py, word in GTable:
            if result.get(word, 0) == 0:
                result[word] = 1
            else:
                result[word] += 1
    if '西医病名.scel' in file_path:#处理的是ICD10诊断.scel，需要先转换为.txt文件
        scel2txt(file_path)#先将scel文件转换为txt文件a
        for count, py, word in GTable:
            if result.get(word, 0) == 0:
                result[word] = 1
            else:
                result[word] += 1
    if '部分疾病名药名.scel' in file_path:#处理的是ICD10诊断.scel，需要先转换为.txt文件
        scel2txt(file_path)#先将scel文件转换为txt文件a
        for count, py, word in GTable:
            if result.get(word, 0) == 0:
                result[word] = 1
            else:
                result[word] += 1
    return result

class dict:#构建一个数据结构，是包装了find方法的字典
    def __init__(self):
        self.dic = {}
    def insert(self, word):
        self.dic[word] = True
    def find(self, word):
        if self.dic.get(word, 0) != 0:
            return True
        else:
            return False

class Trie:#构建一个查找树，更加高效的查找词汇
    # word_end = -1
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = {}
        self.word_end = -1
    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        curNode = self.root
        for c in word:
            if not c in curNode:
                curNode[c] = {}
            curNode = curNode[c]
        curNode[self.word_end] = True
    def find(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        curNode = self.root
        for c in word:
            if not c in curNode:
                return False
            curNode = curNode[c]
        # Doesn't end here
        if self.word_end not in curNode:
            return False
        return True
    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        curNode = self.root
        for c in prefix:
            if not c in curNode:
                return False
            curNode = curNode[c]
        return True

def get_tree_find(file_paths):
    tree = Trie()
    for file_path in file_paths:
        dic = get_data_dic(file_path)
        for word in dic:
            tree.insert(word)
    return tree

def get_dict_find(file_paths):
    dic = dict()
    for file_path in file_paths:
        dic2 = get_data_dic(file_path)
        #print(len(dic2))
        for word in dic2:
            dic.insert(word)
    return dic

if __name__ == '__main__':

    # scel所在文件夹路径
    file_paths = ['./THUOCL_medical.txt', 'body中文身体部位名称.txt', 'disease_new.txt', 'ICD10诊断.scel', 'symptom.txt', '部分疾病名药名.scel', '西医病名.scel', '医院电子病历词库.scel', '症状.scel']
    tree = get_tree_find(file_paths)
    print(tree.find("直肠恶性肿瘤"))
    dict = get_dict_find(file_paths)
    print(dict.find('肠胃炎'))
    print(dict.find('脑膜炎'))
    #print(len(dict), len(tree))

    #file_path = "symptom.txt"
    #dic = get_data_dic(file_path)
    #print(dic)
    #print(len(dic))
