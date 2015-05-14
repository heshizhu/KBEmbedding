#include <stdio.h>
#include <vector>
#include <map>

using namespace std;

class triple{
public:
	long h, r, t;
	triple(long t1, long t2, long t3){
		h = t1; r = t2; t = t3;
	}
	triple(const triple &that){
		h = that.h; r = that.r; t = that.t;
	}
};
typedef vector<triple> formula;

double rand(double min, double max);
double square(double a);
double fast_rev_sqrt(double number);
double fast_sqrt(double number);

double sigmod(double x);
double sigmod_grad(double x);//sigmod(x)的导数

double norm_1(vector<double> &a);
double norm_2(vector<double> &a);
double square_norm_2(vector<double> &a);
double inner(vector<double> &v1, vector<double> &v2);
vector<double> sub(vector<double> &v1, vector<double> &v2);
vector<double> add(vector<double> &v1, vector<double> &v2);
double l1_distance(vector<double> &v1, vector<double> &v2);
double l2_distance(vector<double> &v1, vector<double> &v2);
double square_l2_distance(vector<double> &v1, vector<double> &v2);
double max(vector<double> &v);
double min(vector<double> &v);

//得到vec向量在法线hyper表示的超平面上的映射vec - hyper'*vec*hyper
vector<double> mapHyper(vector<double> &vec, vector<double> &hyper);
//normalize vector and matrix
void normalize(vector<double> &a);
//disorder a list randomly
void random_disorder_list(vector<unsigned> &list);

int ArgPos(char *str, int argc, char **argv);

vector<string> split(const string &text, const string &sep);


//给定正例和负例的值和阈值，返回相应得分
void test(vector<double> &pos_dist, vector<double> &neg_dist, double threshold, double &score);
//给定正例和负例的值，得到阈值及相应得分
void valid(vector<double> &triple_pos_dist, vector<double> &triple_neg_dist, double &threshold, double &score);

void eval_valid(
	vector<triple> &valid_pos, vector<triple> &valid_neg,
	map<unsigned, double> &thre_rels, double &thre_entire,
	double(*loss_triple)(triple tri));

double eval_test(
	vector<triple> &test_pos, vector<triple> &test_neg,
	map<unsigned, double> &thre_rels, double &thre_entire,
	double(*loss_triple)(triple tri));