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
double L1_distance(vector<double> &v1, vector<double> &v2);
double L2_distance(vector<double> &v1, vector<double> &v2);
double square_L2_distance(vector<double> &v1, vector<double> &v2);
double max(vector<double> &v);
double min(vector<double> &v);

//得到ent_vec向量在rel_vec表示的超平面上的映射 ent_vec - rel_vec'*ent_vec*rel_vec
vector<double> transHyper(vector<double> &ent_vec, vector<double> &rel_vec);
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