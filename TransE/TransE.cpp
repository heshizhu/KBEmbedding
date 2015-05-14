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

//string model_base_path = "G:/temp/TransX/fb13/JLeRR_2/";
string model_base_path = "";//currrent path
double loss_sum;

//global parameters
bool L1_Flag = 1;
bool Neg_Scope = 0;
bool Neg_Method = 1;//bern和unif
int  Grad_Method = 2;
int  Batch_Size = 120;
int  Epoch_Size = 500;

int    n = 100;
double rate = 1;
double margin = 0.25;


//global variables
vector<vector<double> > entity_vec, entity_vec_temp;
vector<vector<double> > relation_vec, relation_vec_temp;

//for AdaGrad gradient update
vector<vector<double> > ada_entity_vec, ada_relation_vec;


//origin data
long entity_num, relation_num;
map<string, unsigned> entity2id, relation2id;
map<unsigned, string> id2entity, id2relation;

map<unsigned, map<unsigned, set<unsigned> > > sub_rel_objs;
map<unsigned, vector<unsigned> > rel_heads, rel_tails;
vector<double> head_num_per_tail, tail_num_per_head;//平均每个head有多少个tail, 平均每个tail有多少个head

//train data
long triple_num;
vector<triple> triple_atoms;

/*************** variables and method using for test ***************/
double best_precision_bern = 0, best_precision_unif = 0;

double thre_entire_bern, thre_entire_unif;//全体阈值
map<unsigned, double> thre_rels_bern, thre_rels_unif;//每个关系的阈值

vector<triple> valid_pos_bern, valid_neg_bern, test_pos_bern, test_neg_bern;
vector<triple> valid_pos_unif, valid_neg_unif, test_pos_unif, test_neg_unif;

void eval_loadCorpus();
double evaluation(bool isbern);
/*************** variables and method using for test ***************/

bool exist(triple tri){
	if (sub_rel_objs.count(tri.h) == 0)
		return false;
	if (sub_rel_objs[tri.h].count(tri.r) == 0)
		return false;
	if (sub_rel_objs[tri.h][tri.r].count(tri.t) == 0)
		return false;
	return true;
}

void saveModel(int epoch){
	string version = "unif";
	if (Neg_Method == 1) version = "bern";

	double precision = evaluation(Neg_Method);
	cout << "-------------------------" << endl;
	cout << "test precision(" << version << "): " << precision << endl;
	cout << "best precision(" << version << "): ";
	if (Neg_Method == 1){
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

	FILE* f1 = fopen(("entity2vec." + std::to_string(n) + "." + version).c_str(), "w");
	for (int kk = 0; kk < entity_num; kk++){
		for (int dim = 0; dim < n; dim++)
			fprintf(f1, "%.6lf\t", entity_vec[kk][dim]);
		fprintf(f1, "\n");
	}
	fclose(f1);
	FILE* f2 = fopen(("relation2vec." + std::to_string(n) + "." + version).c_str(), "w");
	for (int kk = 0; kk < relation_num; kk++){
		for (int dim = 0; dim < n; dim++)
			fprintf(f2, "%.6lf\t", relation_vec[kk][dim]);
		fprintf(f2, "\n");
	}
	fclose(f2);
}


void paramater_update(
	map<unsigned, vector<double> > &entity_vec_grad_temp,
	map<unsigned, vector<double> > &relation_vec_grad_temp){

	for (map<unsigned, vector<double> >::iterator it_inner = entity_vec_grad_temp.begin();
		it_inner != entity_vec_grad_temp.end(); it_inner++){
		int ent_id = it_inner->first;
		for (int ii = 0; ii < n; ii++){
			double grad = it_inner->second[ii];
			if (Grad_Method == 2){
				ada_entity_vec[ent_id][ii] += square(grad);
				entity_vec[ent_id][ii] -= (grad * fast_rev_sqrt(ada_entity_vec[ent_id][ii] + EPSILON) * rate);
			}
			else
				entity_vec[ent_id][ii] -= (rate * grad);
		}
		normalize(entity_vec[ent_id]);
	}

	for (map<unsigned, vector<double> >::iterator it_inner = relation_vec_grad_temp.begin();
		it_inner != relation_vec_grad_temp.end(); it_inner++){
		int rel_id = it_inner->first;
		for (int kk = 0; kk < n; kk++){
			double grad = it_inner->second[kk];
			if (Grad_Method == 2){
				ada_relation_vec[rel_id][kk] += square(grad);
				relation_vec[rel_id][kk] -= (grad * fast_rev_sqrt(ada_relation_vec[rel_id][kk] + EPSILON) * rate);
			}
			else
				relation_vec[rel_id][kk] -= (rate * grad);
		}
		normalize(relation_vec[rel_id]);
	}
}

void trainTriple(triple pos_tri, triple neg_tri){
	vector<double> pos_diff = sub(add(entity_vec_temp[pos_tri.h], relation_vec_temp[pos_tri.r]), entity_vec_temp[pos_tri.t]);
	vector<double> neg_diff = sub(add(entity_vec_temp[neg_tri.h], relation_vec_temp[neg_tri.r]), entity_vec_temp[neg_tri.t]);

	double pos_energy, neg_energy;
	if (L1_Flag){
		pos_energy = norm_1(pos_diff);
		neg_energy = norm_1(neg_diff);
	}
	else{
		pos_energy = square_norm_2(pos_diff) / 2;
		neg_energy = square_norm_2(neg_diff) / 2;
	}
	if (pos_energy + margin <= neg_energy) return;

	loss_sum += (pos_energy + margin - neg_energy);

	map<unsigned, vector<double> > entity_vec_grad_temp;
	map<unsigned, vector<double> > relation_vec_grad_temp;
	entity_vec_grad_temp[pos_tri.h].resize(n);
	entity_vec_grad_temp[pos_tri.t].resize(n);
	if (neg_tri.h == pos_tri.h) entity_vec_grad_temp[neg_tri.t].resize(n);
	else entity_vec_grad_temp[neg_tri.h].resize(n);
	relation_vec_grad_temp[pos_tri.r].resize(n);

	//求解梯度
	double loss;
	for (int dd = 0; dd < n; dd++){
		//pos
		loss = pos_diff[dd];
		if (loss != 0){
			if (L1_Flag)
				if (loss > 0) loss = 1; else loss = -1;
			entity_vec_grad_temp[pos_tri.h][dd] += loss;
			relation_vec_grad_temp[pos_tri.r][dd] += loss;
			entity_vec_grad_temp[pos_tri.t][dd] -= loss;
		}		

		//neg
		loss = neg_diff[dd];
		if (loss != 0){
			if (L1_Flag)
				if (loss > 0) loss = 1; else loss = -1;
			loss *= -1;
			entity_vec_grad_temp[neg_tri.h][dd] += loss;
			relation_vec_grad_temp[neg_tri.r][dd] += loss;
			entity_vec_grad_temp[neg_tri.t][dd] -= loss;
		}
	}

	#pragma omp critical
	{
		paramater_update(entity_vec_grad_temp, relation_vec_grad_temp);
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
			if (is_head) neg_tri.h = rand() % entity_num;
			else neg_tri.t = rand() % entity_num;
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
	vector<unsigned> batch_list(triple_num);
	for (int k = 0; k < triple_num; k++) batch_list[k] = k;
	random_disorder_list(batch_list);

	int batchs = triple_num / Batch_Size;//每个batch有batch_size个样本

	for (int bat = 0; bat < batchs; bat++){
		int start = bat * Batch_Size;
		int end = (bat + 1) * Batch_Size;
		if (end > triple_num)
			end = triple_num;

		entity_vec_temp = entity_vec;
		relation_vec_temp = relation_vec;
		#pragma omp parallel for schedule(dynamic) num_threads(THREAD_NUM)
		for (int index = start; index < end; index++)
			trainTriple(triple_atoms[batch_list[index]]);
	}
}

double energy_function(triple tri){
	vector<double> head = add(entity_vec_temp[tri.h], relation_vec_temp[tri.r]);
	if (L1_Flag)
		return l1_distance(head, entity_vec_temp[tri.t]);
	else
		return square_l2_distance(head, entity_vec_temp[tri.t]) / 2;
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

void initModel(){
	entity_vec.resize(entity_num);
	for (int ee = 0; ee < entity_num; ee++){
		entity_vec[ee].resize(n);
		for (int dd = 0; dd < n; dd++)
			entity_vec[ee][dd] = rand(-1, 1);
		normalize(entity_vec[ee]);
	}
	relation_vec.resize(relation_num);
	for (int rr = 0; rr < relation_num; rr++){
		relation_vec[rr].resize(n);
		for (int dd = 0; dd < n; dd++)
			relation_vec[rr][dd] = rand(-1, 1);
		normalize(relation_vec[rr]);
	}
	cout << "init entity vector, relation vector and formula weights are over" << endl;

	//or AdaGrad gradient update, sum of square of every steps
	ada_entity_vec.resize(entity_num);
	for (int kk = 0; kk < entity_num; kk++)
		ada_entity_vec[kk].resize(n, 0);
	ada_relation_vec.resize(relation_num);
	for (int kk = 0; kk < relation_num; kk++)
		ada_relation_vec[kk].resize(n, 0);
	cout << "init adagrad parameters are over" << endl;
}

void loadCorpus(){
	char buf[1000];
	int id;
	FILE *f_ent_id = fopen((model_base_path + "../data/entity2id.txt").c_str(), "r");
	while (fscanf(f_ent_id, "%s%d", buf, &id) == 2){
		string ent = buf; entity2id[ent] = id; id2entity[id] = ent; entity_num++;
	}
	fclose(f_ent_id);
	FILE *f_rel_id = fopen((model_base_path + "../data/relation2id.txt").c_str(), "r");
	while (fscanf(f_rel_id, "%s%d", buf, &id) == 2){
		string rel = buf; relation2id[rel] = id; id2relation[id] = rel; relation_num++;
	}
	fclose(f_rel_id);
	cout << "entity number = " << entity_num << endl;
	cout << "relation number = " << relation_num << endl;

	unsigned sub_id, rel_id, obj_id;
	string line;

	//读取三元组
	ifstream f_kb(model_base_path + "../data/train.txt");
	map<unsigned, set<unsigned> > rel_heads_temp, rel_tails_temp;

	map<unsigned, map<unsigned, set<unsigned> > > relation_head_tails;//计算平均一个head有多少个tail
	map<unsigned, map<unsigned, set<unsigned> > > relation_tail_heads;//计算平均一个tail有多少个head

	while (getline(f_kb, line)){
		vector<string> terms = split(line, "\t");
		sub_id = entity2id[terms[0]]; rel_id = relation2id[terms[1]]; obj_id = entity2id[terms[2]];
		triple_atoms.push_back(triple(sub_id, rel_id, obj_id)); triple_num++;

		sub_rel_objs[sub_id][rel_id].insert(obj_id);
		rel_heads_temp[rel_id].insert(sub_id);
		rel_tails_temp[rel_id].insert(obj_id);
		relation_head_tails[rel_id][sub_id].insert(obj_id);
		relation_tail_heads[rel_id][obj_id].insert(sub_id);
	}
	f_kb.close();
	cout << "tripe number = " << triple_num << endl;

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

	tail_num_per_head.resize(relation_num);
	head_num_per_tail.resize(relation_num);
	for (int rel_id = 0; rel_id < relation_num; rel_id++){
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
	string subject, relation, object;
	while (fscanf(f_kb, "%s", buf) == 1){
		subject = buf;
		fscanf(f_kb, "%s", buf);
		relation = buf;
		fscanf(f_kb, "%s", buf);
		object = buf;
		triple tri(entity2id[subject], relation2id[relation], entity2id[object]);
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
	vector<double> head = add(entity_vec[tri.h], relation_vec[tri.r]);
	vector<double> tail = entity_vec[tri.t];
	if (L1_Flag == 1)
		return l1_distance(head, tail);
	else
		return square_l2_distance(head, tail) / 2;
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


int main(int argc, char**argv){
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

	cout << "L1 / L2 = " << L1_Flag << endl;
	cout << "negative scope = " << Neg_Scope << endl;
	cout << "negative method = " << Neg_Method << endl;
	cout << "grad method = " << Grad_Method << endl;
	cout << "batch = " << Batch_Size << endl;
	cout << "epoch = " << Epoch_Size << endl;
	cout << "dim = " << n << endl;
	cout << "rate = " << rate << endl;
	cout << "margin = " << margin << endl;

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