#include <iostream>
#include <pybind11/pybind11.h>


namespace py = pybind11;


struct TopicWordPair
{
    int t;  // topic assignment
    int d;  // document that this word belongs to
    
    std::string word;
    
    TopicWordPair(std::string word, int doc_id)
        : t(0), d(doc_id), word(word) { }
};

struct TopicWordMatrix
{
    double A;  // hyperparameters
    double B;
    
    int T;  // number of topics
    int W;  // number of unique words in the corpus
    int D;  // number of documents

    int **wt;  // word-topic counts
    int **dt;  // document-topic counts
    int *_t;   // number of words assigned to each topic
    int *d_;   // number of words in each document

    double **term1;
    double **term2;

    py::list docs;
    py::dict words;
    
    TopicWordMatrix(int ntopics, py::list docs, py::dict words)
        : A(0.01), B(0.01), T(ntopics), W(words.size()), D(docs.size()), docs(docs), words(words)
        {
            wt = new int*[W];
            dt = new int*[D];
            
            for (int w = 0; w < W; w++)
                wt[w] = new int[T];
            
            for (int d = 0; d < D; d++)
                dt[d] = new int[T];
            
            _t = new int[T];
            d_ = new int[D];
            
            for (int t = 0; t < T; t++)
            {
                for (int w = 0; w < W; w++)
                    wt[w][t] = 0;
                
                for (int d = 0; d < D; d++)
                    dt[d][t] = 0;
                
                _t[t] = 0;
            }
            
            for (int d = 0; d < D; d++)
                d_[d] = 0;
            
            term1 = new double*[W];
            term2 = new double*[D];
            
            for (int w = 0; w < W; w++)
            {
                term1[w] = new double[T];
                
                for (int t = 0; t < T; t++)
                    term1[w][t] = 0;
            }
            
            for (int d = 0; d < D; d++)
            {
                term2[d] = new double[T];
                
                for (int t = 0; t < T; t++)
                    term2[d][t] = 0;
            }
            
            for (auto doc : docs)
            {
                for (auto obj : doc)
                {
                    TopicWordPair *twPair = obj.cast<TopicWordPair *>();
                    
                    twPair->t = rand() % T;
                    
                    int t = twPair->t;
                    int d = twPair->d;
                    
                    int w = words.attr("__getitem__")(twPair->word).cast<int>();
                    
                    wt[w][t] += 1;
                    dt[d][t] += 1;
                    _t[t] += 1;
                    d_[d] += 1;
                }
            }
            
            for (int w = 0; w < W; w++)
                for (int t = 0; t < T; t++)
                    term1[w][t] = (wt[w][t] + B) / (_t[t] + W * B);
            
            for (int d = 0; d < D; d++)
                for (int t = 0; t < T; t++)
                    term2[d][t] = (dt[d][t] + A);
            
        }
    
    ~TopicWordMatrix()
        {
            for (int w = 0; w < W; w++)
            {
                delete[] wt[w];
                delete[] term1[w];
            }
            
            for (int d = 0; d < D; d++)
            {
                delete[] dt[d];
                delete[] term2[d];
            }
            
            delete[] wt;
            delete[] dt;
            delete[] _t;
            delete[] d_;
            delete[] term1;
            delete[] term2;
        }

    void sweep(int rounds)
        {
            for (int n = 0; n < rounds; n++)
            {
                for (auto doc : docs)
                {
                    for (auto obj : doc)
                    {
                        TopicWordPair *twPair = obj.cast<TopicWordPair *>();
                        
                        makemove(twPair);
                    }
                }
                // go to next round
            }
            // done
        }
    
    void makemove(TopicWordPair *twPair)
        {
            int w = words.attr("__getitem__")(twPair->word).cast<int>();
            
            int t = twPair->t;
            int d = twPair->d;
            
            // Exclude the current word token from the counts
            wt[w][t] -= 1;
            dt[d][t] -= 1;
            _t[t] -= 1;
            d_[d] -= 1;
            
            term1[w][t] = (wt[w][t] + B) / (_t[t] + W * B);
            term2[d][t] = (dt[d][t] + A);
            
            double norm = 0;
            double probs[T];
            
            for (t = 0; t < T; t++)
            {
                probs[t] = term1[w][t] * term2[d][t];
                norm += probs[t];
            }
            
            double r = rand() / (double) RAND_MAX;
            
            for (t = 0; t < T; t++)
            {
                probs[t] /= norm;
                
                if (t != 0)
                    probs[t] += probs[t - 1];
                
                if (r < probs[t])
                    break;
            }
            
            twPair->t = t;
            
            // Add the current word token back to the counts
            wt[w][t] += 1;
            dt[d][t] += 1;
            _t[t] += 1;
            d_[d] += 1;
            
            term1[w][t] = (wt[w][t] + B) / (_t[t] + W * B);
            term2[d][t] = (dt[d][t] + A);
            
            // Finished
        }
    
    py::dict get_topic_word_dist(int t)
        {
            py::dict twdist;
            double denom = 1 / (_t[t] + W * B);
            
            for (auto obj : words)
            {
                std::string word = obj.first.cast<std::string>();
                int w = obj.second.cast<int>();
                
                double prob = (wt[w][t] + B) * denom;

                twdist.attr("__setitem__")(word, prob);
            }
            
            return twdist;
        }

    py::list get_document_topic_dist(int d)
        {
            py::list dtdist;
            double denom = 1 / (d_[d] + T * A);
            
            for (int t = 0; t < T; t++)
            {
                double prob = (dt[d][t] + A) * denom;
                
                dtdist.attr("append")(prob);
            }
            
            return dtdist;
        }
};

PYBIND11_MODULE(Lda, m)
{
    m.doc() = "LDA";
    
    py::class_<TopicWordPair>(m, "TopicWordPair")
        .def(py::init<std::string, int>())
        .def("__repr__",
             [](const TopicWordPair &twp)
                 {
                     return "<" + std::to_string(twp.t) + " '" + twp.word + "'>";
                 }
            )
        .def_readwrite("word", &TopicWordPair::word)
        .def_readwrite("d", &TopicWordPair::d)
        .def_readwrite("t", &TopicWordPair::t);
    
    py::class_<TopicWordMatrix>(m, "TopicWordMatrix")
        .def(py::init<int, py::list, py::dict>())
        .def_readwrite("A", &TopicWordMatrix::A)
        .def_readwrite("B", &TopicWordMatrix::B)
        .def_readwrite("T", &TopicWordMatrix::T)
        .def_readwrite("W", &TopicWordMatrix::W)
        .def_readwrite("D", &TopicWordMatrix::D)
        .def_readwrite("docs", &TopicWordMatrix::docs)
        .def_readwrite("words", &TopicWordMatrix::words)
        .def("sweep", &TopicWordMatrix::sweep)
        .def("get_topic_word_dist", &TopicWordMatrix::get_topic_word_dist)
        .def("get_document_topic_dist", &TopicWordMatrix::get_document_topic_dist);
}
