#include <stdio.h>
#include <vector>
#include <map>

using namespace std;

class triple{
public:
	long h, r, t;//head entity, relation and tail entity
	float conf;//���Ŷ�
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

double norm_2(vector< double> &a);
double inner(vector<double> &v1, vector<double> &v2);
vector<double> sub(vector<double> &v1, vector<double> &v2);
vector<double> add(vector<double> &v1, vector<double> &v2);
double L1_distance(vector<double> &v1, vector<double> &v2);
double L2_distance(vector<double> &v1, vector<double> &v2);
double square_L2_distance(vector<double> &v1, vector<double> &v2);
double max(vector<double> &v);
double min(vector<double> &v);

//�õ�ent_vec������rel_vec��ʾ�ĳ�ƽ���ϵ�ӳ�� ent_vec - rel_vec'*ent_vec*rel_vec
vector<double> transHyper(vector<double> &ent_vec, vector<double> &rel_vec);
void normalize(vector<double> &a);
void normalize2one(vector<double> &a);
void random_disorder_list(vector<unsigned> &list);
int ArgPos(char *str, int argc, char **argv);

//���������͸�����ֵ����ֵ��������Ӧ�÷�
void test(vector<double> &pos_dist, vector<double> &neg_dist, double threshold, double &score);
//���������͸�����ֵ���õ���ֵ����Ӧ�÷�
void valid(vector<double> &triple_pos_dist, vector<double> &triple_neg_dist, double &threshold, double &score);

void eval_valid(vector<triple> &valid_pos, vector<triple> &valid_neg, map<unsigned, double> &thre_rels, double &thre_entire,
	double(*loss_triple)(triple tri));

double eval_test(vector<triple> &test_pos, vector<triple> &test_neg, map<unsigned, double> &thre_rels, double &thre_entire,
	double(*loss_triple)(triple tri));