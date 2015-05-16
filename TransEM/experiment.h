#include <stdio.h>
#include <vector>
#include <map>

using namespace std;

class triple{
public:
	long h, r, t;//head entity, relation and tail entity
	float conf;//置信度
	triple(long t1, long t2, long t3){
		h = t1; r = t2; t = t3; conf = 1.0f;
	}
	triple(long t1, long t2, long t3, float v1){
		h = t1; r = t2; t = t3; conf = v1;
	}
	triple(const triple &that){
		h = that.h; r = that.r; t = that.t; conf = that.conf;
	}
};

double rand(double min, double max);
double square(double a);
double fast_rev_sqrt(double number);
double fast_sqrt(double number);

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

void normalize(vector<double> &a);
void normalize2one(vector<double> &a);

void random_disorder_list(vector<unsigned> &list);
int ArgPos(char *str, int argc, char **argv);
vector<string> split(const string &text, const string &sep);

//三元组分类的测试

//给定正例和负例的值和阈值，返回相应准确率
void test(vector<double> &pos_dist, vector<double> &neg_dist, double threshold, double &score);
//给定正例和负例的值，得到阈值及相应得分
void valid(vector<double> &triple_pos_dist, vector<double> &triple_neg_dist, double &threshold, double &score);


void eval_valid(
	vector<triple> &valid_pos, vector<triple> &valid_neg,
	map<unsigned, double> &thre_rels, double &thre_entire,
	double(*energy_function)(triple tri));

double eval_test(
	vector<triple> &test_pos, vector<triple> &test_neg,
	map<unsigned, double> &thre_rels, double &thre_entire,
	double(*energy_function)(triple tri));