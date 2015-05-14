#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <ctime>
#include <fstream>
#include <cstring>
#include "experi.h"

using namespace std;

typedef unsigned long long uint64_t;
typedef unsigned int uint32_t;

//[min, max)
double rand(double min, double max)
{
	return min + (max - min)*rand() / (RAND_MAX + 1.0);//使用系统默认的随机值
}

//sqrare
double square(double a){
	return a * a;
}

double sigmod(double x){
	return 1.0 / (1.0 + exp(-1 * x));
}

//sigmod(x)的导数
double sigmod_grad(double x){
	return x * (1 - x);
}

//reverse of sqrt 
double fast_rev_sqrt(double number)
{
	uint64_t i;
	double x2, y;
	x2 = number * 0.5;
	y = number;
	i = *(uint64_t *)&y;
	i = 0x5fe6eb50c7b537a9 - (i >> 1);
	y = *(double *)&i;
	y = y * (1.5 - (x2 * y * y));
	y = y * (1.5 - (x2 * y * y));
	return y;
}

//sqrt
double fast_sqrt(double number)
{
	return 1 / fast_rev_sqrt(number);
}

double norm_1(vector< double> &a){
	double norm = 0;
	for (unsigned i = 0; i < a.size(); i++)
		norm += abs(a[i]);
	return norm;
}

double norm_2(vector< double> &a){
	double norm = 0;
	for (unsigned i = 0; i < a.size(); i++)
		norm += (a[i] * a[i]);
	return fast_sqrt(norm);
}

double square_norm_2(vector<double> &a){
	double norm = 0;
	for (unsigned i = 0; i < a.size(); i++)
		norm += (a[i] * a[i]);
	return norm;
}

double inner(vector<double> &v1, vector<double> &v2){
	double value = 0;
	for (unsigned k = 0; k < v1.size(); k++)
		value += v1[k] * v2[k];
	return value;
}

vector<double> sub(vector<double> &v1, vector<double> &v2)
{
	vector<double> values(v1.size(), 0);
	for (unsigned k = 0; k < v1.size(); k++)
		values[k] = v1[k] - v2[k];
	return values;
}

vector<double> add(vector<double> &v1, vector<double> &v2)
{
	vector<double> values(v1.size(), 0);
	for (unsigned k = 0; k < v1.size(); k++)
		values[k] = v1[k] + v2[k];
	return values;
}


double l1_distance(vector<double> &v1, vector<double> &v2)
{
	double sum = 0;
	for (unsigned i = 0; i < v1.size(); i++)
		sum += fabs(v1[i] - v2[i]);
	return sum;
}

double l2_distance(vector<double> &v1, vector<double> &v2)
{
	double sum = 0;
	for (unsigned i = 0; i < v1.size(); i++)
		sum += square(v1[i] - v2[i]);
	return fast_sqrt(sum);
}

//the square of euclidean distance
double square_l2_distance(vector<double> &v1, vector<double> &v2)
{
	double sum = 0;
	for (unsigned i = 0; i < v1.size(); i++)
		sum += square(v1[i] - v2[i]);
	return sum;
}


//得到vec向量在法线hyper表示的超平面上的映射vec - hyper'*vec*hyper
vector<double> mapHyper(vector<double> &vec, vector<double> &hyper){
	int n = vec.size();
	double inner_value = inner(vec, hyper);
	vector<double> values(n, 0);
	for (int k = 0; k < n; k++)
		values[k] = vec[k] - inner_value * hyper[k];
	return values;
}

//normalize vector and matrix
void normalize(vector<double> &a)
{
	double sum = norm_2(a);
	if (sum > 1){
		for (unsigned i = 0; i < a.size(); i++)
			a[i] /= sum;
	}
}

//disorder a list randomly
void random_disorder_list(vector<unsigned> &list)
{
	unsigned size = list.size();
	for (unsigned i = 0; i < size; i++){
		unsigned index = rand() % (size - i) + i;
		if (index == i) continue;
		unsigned temp = list[i];
		list[i] = list[index];
		list[index] = temp;
	}
}

double max(vector<double> &v){
	double max = v[0];
	for (int i = 1; i < v.size(); i++)
		if (v[i] > max)
			max = v[i];
	return max;
}
double min(vector<double> &v){
	double min = v[0];
	for (int i = 1; i < v.size(); i++)
		if (v[i] < min)
			min = v[i];
	return min;
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

//字符串切分
vector<string> split(const string &text, const string &sep) {
	vector<string> tokens;
	int start = 0, end = 0;
	while ((end = text.find(sep, start)) != string::npos) {
		tokens.push_back(text.substr(start, end - start));
		start = end + 1;
	}
	tokens.push_back(text.substr(start));
	return tokens;
}



//三元组分类的测试
//给定正例和负例的值和阈值，返回相应得分
void test(vector<double> &pos_dist, vector<double> &neg_dist, double threshold, double &score){
	int rightSum = 0;
	for (unsigned i = 0; i < pos_dist.size(); i++)
		if (pos_dist[i] <= threshold)
			rightSum++;
	for (unsigned i = 0; i < neg_dist.size(); i++)
		if (neg_dist[i] > threshold)
			rightSum++;
	score = rightSum * 1.0 / (pos_dist.size() + neg_dist.size());
}

//给定正例和负例的值，得到阈值及相应得分
void valid(vector<double> &triple_pos_dist, vector<double> &triple_neg_dist, double &threshold, double &score){
	int max_Int, min_Int;
	double max_1 = max(triple_pos_dist);
	double max_2 = max(triple_neg_dist);
	if (max_1 > max_2)
		max_Int = max_1 * 100;
	else
		max_Int = max_2 * 100;

	double min_1 = min(triple_pos_dist);
	double min_2 = min(triple_neg_dist);
	if (min_1 < min_2)
		min_Int = min_1 * 100;
	else
		min_Int = min_2 * 100;

	double maxScore = 0;
	double maxThre;
	for (int ind = min_Int; ind <= max_Int; ind++){
		double score;
		double thre = ind * 0.01;
		test(triple_pos_dist, triple_neg_dist, thre, score);
		if (score > maxScore){
			maxScore = score;
			maxThre = thre;
		}
	}
	threshold = maxThre;
	score = maxScore;
}


void eval_valid(
	vector<triple> &valid_pos, vector<triple> &valid_neg,
	map<unsigned, double> &thre_rels, double &thre_entire,
	double(*loss_triple)(triple tri)){

	vector<double> triple_pos_dist, triple_neg_dist;
	for (unsigned id = 0; id < valid_pos.size(); id++){
		double dist = loss_triple(valid_pos[id]);
		triple_pos_dist.push_back(dist);
	}
	for (unsigned id = 0; id < valid_neg.size(); id++){
		double dist = loss_triple(valid_neg[id]);
		triple_neg_dist.push_back(dist);
	}

	//得到全体阈值和得分
	double score;
	valid(triple_pos_dist, triple_neg_dist, thre_entire, score);

	//对每个关系进行处理
	set<unsigned> relations;
	for (unsigned i = 0; i < valid_pos.size(); i++)
		relations.insert(valid_pos[i].r);
	//遍历每种关系
	for (set<unsigned>::iterator rel_iter = relations.begin();
		rel_iter != relations.end(); rel_iter++){
		unsigned rel_id = *rel_iter;

		vector<double> triple_pos_dist_rel, triple_neg_dist_rel;
		for (unsigned i = 0; i < valid_pos.size(); i++)
			if (valid_pos[i].r == rel_id)
				triple_pos_dist_rel.push_back(triple_pos_dist[i]);
		for (unsigned i = 0; i < valid_neg.size(); i++)
			if (valid_neg[i].r == rel_id)
				triple_neg_dist_rel.push_back(triple_neg_dist[i]);

		double thre;
		double score;
		valid(triple_pos_dist_rel, triple_neg_dist_rel, thre, score);
		thre_rels[rel_id] = thre;
	}
}

double eval_test(
	vector<triple> &test_pos, vector<triple> &test_neg,
	map<unsigned, double> &thre_rels, double &thre_entire,
	double(*loss_triple)(triple tri)){

	vector<double> triple_pos_dist, triple_neg_dist;
	for (unsigned id = 0; id < test_pos.size(); id++){
		double dist = loss_triple(test_pos[id]);
		triple_pos_dist.push_back(dist);
	}
	for (unsigned id = 0; id < test_neg.size(); id++){
		double dist = loss_triple(test_neg[id]);
		triple_neg_dist.push_back(dist);
	}

	int rightSum = 0;
	for (unsigned id = 0; id < test_pos.size(); id++){
		unsigned rel_id = test_pos[id].r;
		double currthre = thre_entire;
		if (thre_rels.count(rel_id) != 0)
			currthre = thre_rels[rel_id];
		if (triple_pos_dist[id] <= currthre) rightSum++;
	}
	for (unsigned id = 0; id < test_neg.size(); id++){
		unsigned rel_id = test_neg[id].r;
		double currthre = thre_entire;
		if (thre_rels.count(rel_id) != 0)
			currthre = thre_rels[rel_id];
		if (triple_neg_dist[id] >= currthre) rightSum++;
	}
	return rightSum * 1.0 / (triple_pos_dist.size() + triple_neg_dist.size());
}