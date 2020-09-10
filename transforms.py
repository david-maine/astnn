
import numpy as np
import pickle
from pycparser import c_ast, c_parser, c_generator
from pycparser.c_ast import TypeDecl, ID
from gensim.models.word2vec import Word2Vec


def restrict_w2v(w2v, restricted_word_set):
    new_vectors = []
    new_vocab = {}
    new_index2entity = []
    new_vectors_norm = []

    for i in range(len(w2v.vocab)):
        word = w2v.index2entity[i]
        vec = w2v.vectors[i]
        vocab = w2v.vocab[word]
        vec_norm = w2v.vectors_norm[i]
        if word in restricted_word_set:
            vocab.index = len(new_index2entity)
            new_index2entity.append(word)
            new_vocab[word] = vocab
            new_vectors.append(vec)
            new_vectors_norm.append(vec_norm)

    w2v.vocab = new_vocab
    w2v.vectors = np.array(new_vectors)
    w2v.index2entity = np.array(new_index2entity)
    w2v.index2word = np.array(new_index2entity)
    w2v.vectors_norm = np.array(new_vectors_norm)


def get_antonym(word):
    return restricted_word2vec.most_similar(positive=[], negative=[word], topn=1, restrict_vocab=None)[0][0]

class declarationRenamer(c_ast.NodeVisitor):
    def visit_Decl(self, node):
        var_name = node.name
        antonymn = get_antonym(var_name)
        node.name = antonymn
        if type(node.type) is TypeDecl:
            node.type.declname = antonymn
        
class assignmentRenamer(c_ast.NodeVisitor):
    def visit_Assignment(self, node):
        if type(node.lvalue) is ID:
            var_name = node.lvalue.name
            antonymn = get_antonym(var_name)
            node.lvalue.name = antonymn
        if type(node.rvalue) is ID:
            var_name = node.rvalue.name
            antonymn = get_antonym(var_name)
            node.rvalue.name = antonymn
            
class unaryOpRenamer(c_ast.NodeVisitor):
    def visit_UnaryOp(self, node):
        if type(node.expr) is ID:
            var_name = node.expr.name
            antonymn = get_antonym(var_name)
            node.expr.name = antonymn
            
class binaryOpRenamer(c_ast.NodeVisitor):
    def visit_BinaryOp(self, node):
        if type(node.left) is ID:
            var_name = node.left.name
            antonymn = get_antonym(var_name)
            node.left.name = antonymn
        if type(node.right) is ID:
            var_name = node.right.name
            antonymn = get_antonym(var_name)
            node.right.name = antonymn




used_vars = pickle.load( open( "/home/david/projects/university/astnn/var_names.pkl", "rb" ) )
restricted_word2vec = Word2Vec.load("/home/david/projects/university/astnn/data/train/embedding/node_w2v_128").wv
restricted_word2vec.most_similar("a")
restrict_w2v(restricted_word2vec, used_vars)

def rename_vars(ast):
    try:
        declaration_renamer = declarationRenamer()
        assignment_renamer = assignmentRenamer()
        unary_op_renamer = unaryOpRenamer()
        binary_op_renamer = binaryOpRenamer()
            
        declaration_renamer.visit(ast)
        assignment_renamer.visit(ast)
        unary_op_renamer.visit(ast)
        binary_op_renamer.visit(ast)
    except:
        pass

import random

dead_codes = [
    '''
    int main() {
        int alpha;
    }
    ''',
    '''
    int main() {
        int alpha = 0;
        int beta = 5;
        int gamma = alpha + beta;
    }
    ''',
    '''
    int main() {
        const int ALPHA = 10;
        const int BETA = 5;
    }
    ''',
    '''
    int main() {
        int alpha = 0;
        if(false) {
            alpha = 1;
        }
    }
    '''
    ,
    '''
    int main() {
        int alpha = 0;
        if(false) {
            alpha = 1;
        } else {
            alpha = 2;
        }
    }
    ''',
    '''
    int main() {
        int alpha;
    }
    ''',
    '''
    int main() {
        int alpha = 0;
        int beta = 5;
        int gamma = alpha + beta;
    }
    ''',
    '''
    int main() {
        const int ALPHA = 10;
        const int BETA = 5;
    }
    ''',
    '''
    int main() {
        int alpha = 0;
        if(false) {
            alpha = 1;
        }
    }
    '''
    ,
    '''
    int main() {
        int alpha = 0;
        if(false) {
            alpha = 1;
        } else {
            alpha = 2;
        }
    }
    '''
]

parser = c_parser.CParser()

compounds = []
for code in dead_codes:
    ast = parser.parse(code)
    compounds.append(ast.ext[0].body)

class deadCodeAdder(c_ast.NodeVisitor):
    def visit_FuncDef(self, node):
        if node.decl.name == 'main':
            for compound in compounds:
                index = random.randrange(len(node.body.block_items))
                node.body.block_items = node.body.block_items[:index] + compound.block_items + node.body.block_items[index:]

def add_dead_code(ast):
    v = deadCodeAdder()
    v.visit(ast)
    
      