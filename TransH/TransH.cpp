#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <ctime>
#include <cstring>
#include <omp.h>
#include "experi.h"

using namespace std;

#define THREAD_NUM 4//线程个数
double EPSILON = 1e-6;

bool L1_Flag = 0;//包含L1/L2两个版本
int  Grad_Method = 2;//1为SGD, 2为AdaGrad默认为1；今后加入其它梯度更新方法
bool Neg_Scope = 0;//1表示限定关系，0表示全体
bool Neg_Method = 1;//1表示bern, 0表示unif
int  Batch_Size = 120;
int  Epoch_Size = 500;

int    n = 50; //dimension of entity and relation
double rate = 1;//当使用AdaGrad的时候设置为1
double margin = 1;
double c_value = 0.0625;//实体归一化权重

//global variables
vector<vector<double> > ent_vec, ent_vec_temp; //entity vector
vector<vector<double> > rel_vec, rel_vec_temp; //relation vector
vector<vector<double> > rel_hyper, rel_hyper_temp; //relation vector

//for AdaGrad gradient update, sum of square of every steps
vector<vector<double> > ada_ent_vec, ada_rel_vec, ada_rel_hyper;

//origin data
long ent_num, rel_num;
map<string, unsigned> ent2id, rel2id;
map<unsigned, string> id2ent, id2rel;

//train data
long tri_num;
vector<triple> tri_atoms;

map<unsigned, map<unsigned, set<unsigned> > > sub_rel_objs;
map<unsigned, vector<unsigned> > rel_heads, rel_tails;//the head and tails with relation
vector<double> head_num_per_tail, tail_num_per_head;//平均每个head有多少个tail, 平均每个tail有多少个head

string version;
string dim_str;
double loss_sum = 0;

//string model_base_path = "G:/temp/TransX/fb13/TransH/";
string model_base_path = "";//currrent path

/*************** variables and method using for test ***************/
double best_precision_bern = 0, best_precision_unif = 0;

double thre_entire_bern, thre_entire_unif;//全体阈值
map<unsigned, double> thre_rels_bern, thre_rels_unif;//每个关系的阈值

vector<triple> valid_pos_bern, valid_neg_bern, test_pos_bern, test_neg_bern;
vector<triple> valid_pos_unif, valid_neg_unif, test_pos_unif, test_neg_unif;

void eval_loadCorpus();
double evaluation(bool isbern);
/*************** variables and method using for test ***************/

void saveModel(int epoch){
	double precision = evaluation(Neg_Method);
	cout << "-------------------------" << endl;
	cout << "test precision(" << version << "): " << precision << endl;
	cout << "best precision(" << version << "): ";
	if (Neg_Method){
		cout << best_precision_bern << endl;
		if (precision > best_precision_bern)
			best_precision_bern = precision;
		else return;
	}
	else{
		cout << best_precision_unif << endl;
		if (precision > best_precision_unif)
			best_precision_unif = precision;
		else return;
	}
	FILE* f1 = fopen(("entity2vec." + dim_str + "." + version).c_str(), "w");
	for (int kk = 0; kk < ent_num; kk++){
		for (int dim = 0; dim < n; dim++)
			fprintf(f1, "%.6lf\t", ent_vec[kk][dim]);
		fprintf(f1, "\n");
	}
	fclose(f1);
	FILE* f2 = fopen(("relation2vec." + dim_str + "." + version).c_str(), "w");
	for (int kk = 0; kk < rel_num; kk++){
		for (int dim = 0; dim < n; dim++)
			fprintf(f2, "%.6lf\t", rel_vec[kk][dim]);
		fprintf(f2, "\n");
	}
	fclose(f2);
	FILE* f3 = fopen(("relation2hyper." + dim_str + "." + version).c_str(), "w");
	for (int kk = 0; kk < rel_num; kk++){
		for (int dim = 0; dim < n; dim++)
			fprintf(f3, "%.6lf\t", rel_hyper[kk][dim]);
		fprintf(f3, "\n");
	}
	fclose(f3);
}

bool exist(triple &tri){
	if (sub_rel_objs.count(tri.h) == 0)
		return false;
	if (sub_rel_objs[tri.h].count(tri.r) == 0)
		return false;
	if (sub_rel_objs[tri.h][tri.r].count(tri.t) == 0)
		return false;
	return true;
}

//update the paramaters, include the AdaGrad gradient and vector reprentation
void paramater_update(
	map<unsigned, vector<double> > &ent_vec_grad_temp,
	map<unsigned, vector<double> > &rel_vec_grad_temp,
	map<unsigned, vector<double> > &rel_hyper_grad_temp){
	
	for (map<unsigned, vector<double> >::iterator it_inner = ent_vec_grad_temp.begin(); 
		it_inner != ent_vec_grad_temp.end(); it_inner++){
		int ent_id = it_inner->first;
		for (int ii = 0; ii < n; ii++){
			double grad = it_inner->second[ii];
			if (Grad_Method == 2){
				ada_ent_vec[ent_id][ii] += square(grad);
				ent_vec[ent_id][ii] -= (grad * fast_rev_sqrt(ada_ent_vec[ent_id][ii] + EPSILON) * rate);
			}
			else ent_vec[ent_id][ii] -= (rate * grad);
		}
		//normalize(ent_vec[ent_id]);
	}
	for (map<unsigned, vector<double> >::iterator it_inner = rel_vec_grad_temp.begin();
		it_inner != rel_vec_grad_temp.end(); it_inner++){
		int rel_id = it_inner->first;
		for (int kk = 0; kk < n; kk++){
			double grad = it_inner->second[kk];
			if (Grad_Method == 2){
				ada_rel_vec[rel_id][kk] += square(grad);
				rel_vec[rel_id][kk] -= (grad * fast_rev_sqrt(ada_rel_vec[rel_id][kk] + EPSILON) * rate);
			}
			else rel_vec[rel_id][kk] -= (rate * grad);
		}
	}
	for (map<unsigned, vector<double> >::iterator it_inner = rel_hyper_grad_temp.begin();
		it_inner != rel_hyper_grad_temp.end(); it_inner++){
		int rel_id = it_inner->first;
		for (int kk = 0; kk < n; kk++){
			double grad = it_inner->second[kk];
			if (Grad_Method == 2){
				ada_rel_hyper[rel_id][kk] += square(grad);
				rel_hyper[rel_id][kk] -= (grad * fast_rev_sqrt(ada_rel_hyper[rel_id][kk] + EPSILON) * rate);
			}
			else rel_hyper[rel_id][kk] -= (rate * grad);
		}
		normalize2one(rel_hyper[rel_id]);//对关系法向量进行归一化
	}
}


void trainTriple(triple pos_tri, triple neg_tri){
	vector<double> pos_h_sub = mapHyper(ent_vec_temp[pos_tri.h], rel_hyper_temp[pos_tri.r]);
	vector<double> pos_h_obj = mapHyper(ent_vec_temp[pos_tri.t], rel_hyper_temp[pos_tri.r]);

	vector<double> neg_h_sub = mapHyper(ent_vec_temp[neg_tri.h], rel_hyper_temp[neg_tri.r]);
	vector<double> neg_h_obj = mapHyper(ent_vec_temp[neg_tri.t], rel_hyper_temp[neg_tri.r]);

	vector<double> pos_h_head = add(pos_h_sub, rel_vec_temp[pos_tri.r]);
	vector<double> neg_h_head = add(neg_h_sub, rel_vec_temp[neg_tri.r]);

	double pos_energy, neg_energy;
	if (L1_Flag){
		pos_energy = l1_distance(pos_h_head, pos_h_obj);
		neg_energy = l1_distance(neg_h_head, neg_h_obj);
	}
	else{
		pos_energy = square_l2_distance(pos_h_head, pos_h_obj) / 2;
		neg_energy = square_l2_distance(neg_h_head, neg_h_obj) / 2;
	}
	vector<unsigned> entities;
	map<unsigned, vector<double> > ent_vec_grad_temp;
	map<unsigned, vector<double> > rel_vec_grad_temp;
	map<unsigned, vector<double> > rel_hyper_grad_temp;
	ent_vec_grad_temp[pos_tri.h].resize(n);
	ent_vec_grad_temp[pos_tri.t].resize(n);
	entities.push_back(pos_tri.h); entities.push_back(pos_tri.t);
	if (neg_tri.h == pos_tri.h){
		ent_vec_grad_temp[neg_tri.t].resize(n);
		entities.push_back(neg_tri.t);
	}
	else{
		ent_vec_grad_temp[neg_tri.h].resize(n);
		entities.push_back(neg_tri.h);
	}
	rel_vec_grad_temp[pos_tri.r].resize(n);
	rel_hyper_grad_temp[pos_tri.r].resize(n);
	//求解梯度
	//第一部分
	double base_loss = pos_energy + margin - neg_energy;
	if (base_loss > 0){
		loss_sum += base_loss;
		//pos部分
		vector<double> pos_diff = sub(pos_h_head, pos_h_obj);
		if (L1_Flag){
			for (int dd = 0; dd < n; dd++){
				if (pos_diff[dd] > 0) pos_diff[dd] = 1; 
				else if (pos_diff[dd] < 0) pos_diff[dd] = -1;
			}
		}
		double pos_head_inner = inner(rel_hyper_temp[pos_tri.r], ent_vec_temp[pos_tri.h]);
		double pos_tail_inner = inner(rel_hyper_temp[pos_tri.r], ent_vec_temp[pos_tri.t]);
		double pos_d_w = 0;
		for (int dd = 0; dd < n; dd++)
			pos_d_w += pos_diff[dd] * rel_hyper[pos_tri.r][dd];
		for (int dd = 0; dd < n; dd++){
			double temp;
			//sub
			temp = pos_diff[dd] - pos_d_w * rel_hyper_temp[pos_tri.r][dd];
			ent_vec_grad_temp[pos_tri.h][dd] += temp;
			//obj
			ent_vec_grad_temp[pos_tri.t][dd] -= temp;
			//rel_vec
			rel_vec_grad_temp[pos_tri.r][dd] += pos_diff[dd];
			//rel_hyper
			temp = pos_diff[dd] * (pos_tail_inner - pos_head_inner) + 
				pos_d_w * (ent_vec_temp[pos_tri.t][dd] - ent_vec_temp[pos_tri.h][dd]);
			rel_hyper_grad_temp[pos_tri.r][dd] += temp;
		}

		//neg部分
		vector<double> neg_diff = sub(neg_h_head, neg_h_obj);
		for (int dd = 0; dd < n; dd++){
			if (L1_Flag){
				if (neg_diff[dd] > 0) neg_diff[dd] = 1;
				else if (neg_diff[dd] < 0) neg_diff[dd] = -1;
			}
			neg_diff[dd] *= -1;
		}
		double neg_head_inner = inner(rel_hyper_temp[neg_tri.r], ent_vec_temp[neg_tri.h]);
		double neg_tail_inner = inner(rel_hyper_temp[neg_tri.r], ent_vec_temp[neg_tri.t]);
		double neg_d_w = 0;
		for (int dd = 0; dd < n; dd++)
			neg_d_w += neg_diff[dd] * rel_hyper[neg_tri.r][dd];
		for (int dd = 0; dd < n; dd++){
			double temp;
			//sub
			temp = neg_diff[dd] - neg_d_w * rel_hyper_temp[neg_tri.r][dd];
			ent_vec_grad_temp[neg_tri.h][dd] += temp;
			//obj
			ent_vec_grad_temp[neg_tri.t][dd] -= temp;
			//rel_vec
			rel_vec_grad_temp[neg_tri.r][dd] += neg_diff[dd];
			//rel_hyper
			temp = neg_diff[dd] * (neg_tail_inner - neg_head_inner) +
				neg_d_w * (ent_vec_temp[neg_tri.t][dd] - ent_vec_temp[neg_tri.h][dd]);
			rel_hyper_grad_temp[neg_tri.r][dd] += temp;
		}
	}
	//实体部分
	for (int kk = 0; kk < entities.size(); kk++){
		unsigned ent_id = entities[kk];
		base_loss = square_norm_2(ent_vec_temp[ent_id]) - 1;
		if (base_loss > 0){
			loss_sum += c_value * base_loss;
			for (int dd = 0; dd < n; dd++)
				ent_vec_grad_temp[ent_id][dd] += 2 * c_value * ent_vec_temp[ent_id][dd];
		}
	}
	//关系法向量与向量部分
	double rel_inner = inner(rel_hyper_temp[pos_tri.r], rel_vec_temp[pos_tri.r]);
	double rel_vec_norm2 = square_norm_2(rel_vec_temp[pos_tri.r]);
	base_loss = square(rel_inner) - EPSILON * rel_vec_norm2;
	if (base_loss > 0 && rel_vec_norm2 > 0){
		loss_sum += c_value * base_loss;
		
		double loss_one = c_value * 2 * rel_inner / rel_vec_norm2;
		double loss_two = c_value * 2 * square(rel_inner / rel_vec_norm2);
		
		for (int dd = 0; dd < n; dd++){
			rel_hyper_grad_temp[pos_tri.r][dd] += loss_one * rel_vec_temp[pos_tri.r][dd];			
			rel_vec_grad_temp[pos_tri.r][dd] += loss_one * rel_hyper_temp[pos_tri.r][dd];
			rel_vec_grad_temp[pos_tri.r][dd] -= loss_two * rel_vec_temp[pos_tri.r][dd];
		}
	}
	
	#pragma omp critical
	{
		paramater_update(ent_vec_grad_temp, rel_vec_grad_temp, rel_hyper_grad_temp);
	}
}

triple sampleNegTriple(triple pos_tri, bool is_head){
	triple neg_tri(pos_tri);
	bool in_relation = Neg_Scope;//是否在关系中选择
	int loop_size = 0;
	while (1){
		if (in_relation){
			if (is_head) neg_tri.h = rel_heads[neg_tri.r][rand() % rel_heads[neg_tri.r].size()];
			else neg_tri.t = rel_tails[neg_tri.r][rand() % rel_tails[neg_tri.r].size()];
		}
		else{
			if (is_head) neg_tri.h = rand() % ent_num;
			else neg_tri.t = rand() % ent_num;
		}
		if (!exist(neg_tri)) break;
		else if (loop_size++ > 10) in_relation = 0;//连续10次收不到，则在全局中抽
	}
	return neg_tri;
}

void trainTriple(triple pos_tri){
	int head_pro = 500;//选择调换head作为负样本的概率
	if (Neg_Method){//bern
		double tph = tail_num_per_head[pos_tri.r];
		double hpt = head_num_per_tail[pos_tri.r];
		head_pro = 1000 * tph / (tph + hpt);
	}
	bool is_head = false;
	if ((rand() % 1000) < head_pro)
		is_head = true;
	trainTriple(pos_tri, sampleNegTriple(pos_tri, is_head));
}

void trainTriple(){
	//random select batch，0 - triple_num
	vector<unsigned> batch_list(tri_num);
	for (int k = 0; k < tri_num; k++) batch_list[k] = k;
	random_disorder_list(batch_list);

	int batchs = tri_num / Batch_Size;//每个batch有batch_size个样本

	for (int bat = 0; bat < batchs; bat++){
		int start = bat * Batch_Size;
		int end = (bat + 1) * Batch_Size;
		if (end > tri_num)
			end = tri_num;
		ent_vec_temp = ent_vec;
		rel_vec_temp = rel_vec;
		rel_hyper_temp = rel_hyper;
		#pragma omp parallel for schedule(dynamic) num_threads(THREAD_NUM)
		for (int index = start; index < end; index++)
			trainTriple(tri_atoms[batch_list[index]]);
	}
}

void trainModel(){
	time_t lt;

	for (int epoch = 0; epoch < Epoch_Size; epoch++){
		lt = time(NULL);
		cout << "*************************" << endl;
		cout << "epoch " << epoch << " begin at: " << ctime(&lt);
		double last_loss_sum = loss_sum;
		loss_sum = 0;

		trainTriple();//基于三元组的约束

		lt = time(NULL);
		cout << "epoch " << epoch << " over  at: " << ctime(&lt);
		cout << "last loss sum : " << last_loss_sum << endl;
		cout << "this loss sum : " << loss_sum << endl;
		cout << "*************************" << endl;
		saveModel(epoch);
	}
}


/***************** 初始化神经网络 ***********************/
void initModel()
{
	ent_vec.resize(ent_num);
	for (unsigned kk = 0; kk < ent_num; kk++){
		ent_vec[kk].resize(n);
		for (int dd = 0; dd < n; dd++)
			ent_vec[kk][dd] = rand(-1, 1);
		normalize2one(ent_vec[kk]);
	}

	rel_vec.resize(rel_num);
	for (int kk = 0; kk < rel_num; kk++){
		rel_vec[kk].resize(n);
		for (int dd = 0; dd < n; dd++)
			rel_vec[kk][dd] = rand(-1, 1);
		normalize2one(rel_vec[kk]);
	}

	rel_hyper.resize(rel_num);
	for (int kk = 0; kk < rel_num; kk++){
		rel_hyper[kk].resize(n);
		for (int dd = 0; dd < n; dd++)
			rel_hyper[kk][dd] = rand(-1, 1);
		normalize2one(rel_hyper[kk]);
	}
	
	cout << "init entity vector and relation matrix is over." << endl;

	//or AdaGrad gradient update, sum of square of every steps
	ada_ent_vec.resize(ent_num);
	for (int kk = 0; kk < ent_num; kk++)
		ada_ent_vec[kk].resize(n, 0);
	ada_rel_vec.resize(rel_num);
	ada_rel_hyper.resize(rel_num);
	for (int kk = 0; kk < rel_num; kk++){
		ada_rel_vec[kk].resize(n, 0);
		ada_rel_hyper[kk].resize(n, 0);
	}
	cout << "init entity adagrad vector and relation matrix adagrad is over." << endl;
}

void loadCorpus(){
	char buf[1000];
	int id;
	FILE *f_ent_id = fopen((model_base_path + "../data/entity2id.txt").c_str(), "r");
	while (fscanf(f_ent_id, "%s%d", buf, &id) == 2){
		string ent = buf; ent2id[ent] = id; id2ent[id] = ent; ent_num++;
	}
	fclose(f_ent_id);

	FILE *f_rel_id = fopen((model_base_path + "../data/relation2id.txt").c_str(), "r");
	while (fscanf(f_rel_id, "%s%d", buf, &id) == 2){
		string rel = buf; rel2id[rel] = id; id2rel[id] = rel; rel_num++;
	}
	fclose(f_rel_id);
	cout << "entity number = " << ent_num << endl;
	cout << "relation number = " << rel_num << endl;

	unsigned sub_id, rel_id, obj_id;
	string line;
	//读取三元组
	ifstream f_kb(model_base_path + "../data/train.txt");
	map<unsigned, set<unsigned> > rel_heads_temp, rel_tails_temp;

	map<unsigned, map<unsigned, set<unsigned> > > relation_head_tails;//计算平均一个head有多少个tail
	map<unsigned, map<unsigned, set<unsigned> > > relation_tail_heads;//计算平均一个tail有多少个head

	while (getline(f_kb, line)){
		vector<string> terms = split(line, "\t");
		sub_id = ent2id[terms[0]]; rel_id = rel2id[terms[1]]; obj_id = ent2id[terms[2]];
		tri_atoms.push_back(triple(sub_id, rel_id, obj_id)); tri_num++;

		sub_rel_objs[sub_id][rel_id].insert(obj_id);
		rel_heads_temp[rel_id].insert(sub_id);
		rel_tails_temp[rel_id].insert(obj_id);
		relation_head_tails[rel_id][sub_id].insert(obj_id);
		relation_tail_heads[rel_id][obj_id].insert(sub_id);
	}
	f_kb.close();
	cout << "tripe number = " << tri_num << endl;

	for (map<unsigned, set<unsigned> >::iterator iter = rel_heads_temp.begin(); iter != rel_heads_temp.end(); iter++){
		unsigned rel_id = iter->first;
		for (set<unsigned>::iterator inner_iter = iter->second.begin(); inner_iter != iter->second.end(); inner_iter++)
			rel_heads[rel_id].push_back(*inner_iter);
	}
	for (map<unsigned, set<unsigned> >::iterator iter = rel_tails_temp.begin(); iter != rel_tails_temp.end(); iter++){
		unsigned rel_id = iter->first;
		for (set<unsigned>::iterator inner_iter = iter->second.begin(); inner_iter != iter->second.end(); inner_iter++)
			rel_tails[rel_id].push_back(*inner_iter);
	}

	tail_num_per_head.resize(rel_num);
	head_num_per_tail.resize(rel_num);
	for (int rel_id = 0; rel_id < rel_num; rel_id++){
		//计算平均一个head有多少个tail
		map<unsigned, set<unsigned> > tails_per_head = relation_head_tails[rel_id];
		unsigned head_number = 0, tail_count = 0;
		for (map<unsigned, set<unsigned> > ::iterator iter = tails_per_head.begin(); iter != tails_per_head.end(); iter++){
			if (iter->second.size() > 0){ head_number++; tail_count += iter->second.size(); }
		}
		tail_num_per_head[rel_id] = 1.0 * tail_count / head_number;
		//计算平均一个tail有多少个head
		map<unsigned, set<unsigned> > heads_per_tail = relation_tail_heads[rel_id];
		unsigned tail_number = 0, head_count = 0;
		for (map<unsigned, set<unsigned> > ::iterator iter = heads_per_tail.begin();
			iter != heads_per_tail.end(); iter++){
			if (iter->second.size() > 0){ tail_number++; head_count += iter->second.size(); }
		}
		head_num_per_tail[rel_id] = 1.0 * head_count / tail_number;
	}

	eval_loadCorpus();
}


/********** begin of triple classification **********/
void load_eval_data(bool isvalid, bool ispos, bool isbern, vector<triple > &triple_list){
	string filename;
	if (isvalid) filename = "valid_"; else filename = "test_";
	if (ispos) filename += "pos_"; else filename += "neg_";
	if (isbern) filename += "bern"; else filename += "unif";

	char buf[1000];
	FILE *f_kb = fopen((model_base_path + "../data/" + filename + ".txt").c_str(), "r");
	string sub, rel, obj;
	while (fscanf(f_kb, "%s", buf) == 1){
		sub = buf; fscanf(f_kb, "%s", buf); rel = buf; fscanf(f_kb, "%s", buf); obj = buf;
		triple tri(ent2id[sub], rel2id[rel], ent2id[obj]);
		triple_list.push_back(tri);
	}
	fclose(f_kb);
	cout << "load: " << filename << endl;
}


void eval_loadCorpus(){
	if (Neg_Method){
		load_eval_data(true, true, true, valid_pos_bern);
		load_eval_data(true, false, true, valid_neg_bern);
		load_eval_data(false, true, true, test_pos_bern);
		load_eval_data(false, false, true, test_neg_bern);
	}
	else{
		load_eval_data(true, true, false, valid_pos_unif);
		load_eval_data(true, false, false, valid_neg_unif);
		load_eval_data(false, true, false, test_pos_unif);
		load_eval_data(false, false, false, test_neg_unif);
	}
}

double loss_triple(triple tri){
	vector<double> sub_h = mapHyper(ent_vec[tri.h], rel_hyper[tri.r]);
	vector<double> head = add(sub_h, rel_vec[tri.r]);
	vector<double> tail = mapHyper(ent_vec[tri.t], rel_hyper[tri.r]);
	if (L1_Flag == 1) return l1_distance(head, tail);
	else return square_l2_distance(head, tail) / 2;
}


void eval_valid(bool isbern){
	if (isbern){
		thre_entire_bern = 0;
		thre_rels_bern.clear();
		eval_valid(valid_pos_bern, valid_neg_bern,
			thre_rels_bern, thre_entire_bern, loss_triple);
	}
	else{
		thre_entire_unif = 0;
		thre_rels_unif.clear();
		eval_valid(valid_pos_unif, valid_neg_unif,
			thre_rels_unif, thre_entire_unif, loss_triple);
	}
}

double eval_test(bool isbern){
	if (isbern)
		return eval_test(test_pos_bern, test_neg_bern,
		thre_rels_bern, thre_entire_bern, loss_triple);
	else
		return eval_test(test_pos_unif, test_neg_unif,
		thre_rels_unif, thre_entire_unif, loss_triple);
}

double evaluation(bool isbern){
	eval_valid(isbern);
	double test_score = eval_test(isbern);
	return test_score;
}
/********** end of triple classification **********/


int main(int argc, char**argv)
{
	int i;
	if ((i = ArgPos((char *)"-l1", argc, argv)) > 0) L1_Flag = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-negScope", argc, argv)) > 0) Neg_Scope = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-negMethod", argc, argv)) > 0) Neg_Method = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-grad", argc, argv)) > 0) Grad_Method = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-batch", argc, argv)) > 0) Batch_Size = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-epoch", argc, argv)) > 0) Epoch_Size = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) n = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-rate", argc, argv)) > 0) rate = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-margin", argc, argv)) > 0) margin = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-c", argc, argv)) > 0) c_value = atof(argv[i + 1]);

	cout << "L1 / L2 = " << L1_Flag << endl;
	cout << "negative scope = " << Neg_Scope << endl;
	cout << "negative method = " << Neg_Method << endl;
	cout << "grad method = " << Grad_Method << endl;
	cout << "batch = " << Batch_Size << endl;
	cout << "epoch = " << Epoch_Size << endl;
	cout << "dim = " << n << endl;
	cout << "rate = " << rate << endl;
	cout << "margin = " << margin << endl;
	cout << "c = " << c_value << endl;
	
	if (Neg_Method) version = "bern"; else version = "unif";
	dim_str = std::to_string(n);

	time_t lt = time(NULL);
	cout << "begin at: " << ctime(&lt);
	loadCorpus();

	lt = time(NULL);
	cout << "prepare over at: " << ctime(&lt);
	initModel();

	lt = time(NULL);
	cout << "init net over at: " << ctime(&lt);

	trainModel();
	lt = time(NULL);
	cout << "train over at: " << ctime(&lt);

	return 1;
}



//void valid_grad(){
//
//	//校验梯度
//	//存储DELTA变化使用
//	map<unsigned, vector<double> > ent_vec_delta;
//	map<unsigned, vector<double> > rel_vec_delta;
//	map<unsigned, vector<double> > rel_hyper_delta;
//	//存储梯度
//	map<unsigned, vector<double> > ent_vec_grad = ent_vec_grad_temp;
//	map<unsigned, vector<double> > rel_vec_grad = rel_vec_grad_temp;
//	map<unsigned, vector<double> > rel_hyper_grad = rel_hyper_grad_temp;
//
//	double delta = 0.0001;
//
//	//实体的变化
//	for (int i = 0; i < entities.size(); i++){
//		unsigned ent_id = entities[i];
//		ent_vec_delta[ent_id].resize(n);
//		for (int dd = 0; dd < n; dd++){
//			double value_src = ent_vec_temp[ent_id][dd];
//
//			ent_vec_temp[ent_id][dd] = value_src + delta;
//			double loss_before = loss_function(pos_tri, neg_tri);
//			ent_vec_temp[ent_id][dd] = value_src - delta;
//			double loss_after = loss_function(pos_tri, neg_tri);
//
//			ent_vec_delta[ent_id][dd] = (loss_before - loss_after) / delta / 2;
//
//			ent_vec_temp[ent_id][dd] = value_src;
//		}
//	}
//	unsigned rel = pos_tri.r;
//	//关系的变化
//	rel_vec_delta[rel].resize(n, 0);
//	for (int dd = 0; dd < n; dd++){
//		double value_src = rel_vec_temp[rel][dd];
//		rel_vec_temp[rel][dd] = value_src + delta;
//		double loss_before = loss_function(pos_tri, neg_tri);
//		rel_vec_temp[rel][dd] = value_src - delta;
//		double loss_after = loss_function(pos_tri, neg_tri);
//		rel_vec_delta[rel][dd] = (loss_before - loss_after) / delta / 2;//变化率	
//		rel_vec_temp[rel][dd] = value_src;//恢复
//	}
//	rel_hyper_delta[rel].resize(n, 0);
//	for (int dd = 0; dd < n; dd++){
//		double value_src = rel_hyper_temp[rel][dd];
//		rel_hyper_temp[rel][dd] = value_src + delta;
//		double loss_before = loss_function(pos_tri, neg_tri);
//		rel_hyper_temp[rel][dd] = value_src - delta;
//		double loss_after = loss_function(pos_tri, neg_tri);
//		rel_hyper_delta[rel][dd] = (loss_before - loss_after) / delta / 2;//变化率	
//		rel_hyper_temp[rel][dd] = value_src;//恢复
//	}
//
//	//验证delta与梯度的差值
//	for (int i = 0; i < entities.size(); i++){
//		unsigned ent_id = entities[i];
//
//		vector<double> deltas = ent_vec_delta[ent_id];
//		vector<double> grads = ent_vec_grad[ent_id];
//
//		cout << "实体: " << ent_id << endl;
//		for (int ii = 0; ii < n; ii++)
//			cout << "\t" << fabs(deltas[ii] - grads[ii]) << "[" << deltas[ii] << "] [" << grads[ii] << "]" << endl;
//	}
//
//	vector<double> vec_deltas = rel_vec_delta[rel];
//	vector<double> vec_grads = rel_vec_grad[rel];
//	cout << "关系(vec): " << rel << endl;
//	for (int kk = 0; kk < n; kk++)
//		cout << "\t" << fabs(vec_deltas[kk] - vec_grads[kk]) << "[" << vec_deltas[kk] << "] [" << vec_grads[kk] << "]" << endl;
//
//	vector<double> hyper_deltas = rel_hyper_delta[rel];
//	vector<double> hyper_grads = rel_hyper_grad[rel];
//	cout << "关系(hyper): " << rel << endl;
//	for (int kk = 0; kk < n; kk++)
//		cout << "\t" << fabs(hyper_deltas[kk] - hyper_grads[kk]) << "[" << hyper_deltas[kk] << "] [" << hyper_grads[kk] << "]" << endl;
//
//
//	exit(1);
//}

//double loss_function(triple pos_tri, triple neg_tri){
//	double loss_temp = 0;
//	vector<double> pos_h_sub = mapHyper(ent_vec_temp[pos_tri.h], rel_hyper_temp[pos_tri.r]);
//	vector<double> pos_h_obj = mapHyper(ent_vec_temp[pos_tri.t], rel_hyper_temp[pos_tri.r]);
//
//	vector<double> neg_h_sub = mapHyper(ent_vec_temp[neg_tri.h], rel_hyper_temp[neg_tri.r]);
//	vector<double> neg_h_obj = mapHyper(ent_vec_temp[neg_tri.t], rel_hyper_temp[neg_tri.r]);
//
//	vector<double> pos_h_head = add(pos_h_sub, rel_vec_temp[pos_tri.r]);
//	vector<double> neg_h_head = add(neg_h_sub, rel_vec_temp[neg_tri.r]);
//
//	double pos_energy, neg_energy;
//	if (L1_Flag){
//		pos_energy = l1_distance(pos_h_head, pos_h_obj);
//		neg_energy = l1_distance(neg_h_head, neg_h_obj);
//	}
//	else{
//		pos_energy = square_l2_distance(pos_h_head, pos_h_obj) / 2;
//		neg_energy = square_l2_distance(neg_h_head, neg_h_obj) / 2;
//	}
//	vector<unsigned> entities;
//	entities.push_back(pos_tri.h); entities.push_back(pos_tri.t);
//	if (neg_tri.h == pos_tri.h)
//		entities.push_back(neg_tri.t);
//	else
//		entities.push_back(neg_tri.h);
//	double score = pos_energy + margin - neg_energy;
//	if (score > 0)
//		loss_temp += score;
//
//	//实体部分
//	for (int kk = 0; kk < entities.size(); kk++){
//		unsigned ent_id = entities[kk];
//		score = square_norm_2(ent_vec_temp[ent_id]) - 1;
//		if (score > 0)
//			loss_temp += c_value * score;
//	}
//
//	//最后一部分
//	unsigned rel = pos_tri.r;
//	score = square(inner(rel_hyper_temp[rel], rel_vec_temp[rel])) / square_norm_2(rel_vec_temp[rel]) - EPSILON;
//	if (score > 0)
//		loss_temp += c_value * score;
//
//	return loss_temp;
//}