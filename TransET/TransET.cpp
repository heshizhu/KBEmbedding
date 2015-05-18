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
#include <sstream>
#include <omp.h>
#include "experiment.h"

using namespace std;

#define THREAD_NUM 64//线程个数
double EPSILON = 1e-6;

//string model_base_path = "G:/temp/TransX/fb15k+Types/TransET/";//currrent path
string model_base_path = "";//currrent path
double loss_sum;

//全局参数
bool L1_Flag = 0;
bool Neg_Scope = 1;//0:global; 1:relation
int  Neg_Method = 2;//1:unif; 2:bern
int  Batch_Size = 2480;//如果为0表示不采用mini-batch
int  Epoch_Size = 1000;

int    n = 50;
double margin = 1;
double rate = 0.001;
string dim_str;
string version;

//embedding
vector<vector<double> > ent_vec, ent_vec_tmp;
vector<vector<double> > rel_vec, rel_vec_tmp;
vector<vector<double> > type_vec, type_vec_tmp;
//vector<vector<vector<double> > > trans_mat, trans_mat_tmp;//转换矩阵

//origin data
long ent_num, rel_num, type_num;
map<string, unsigned> ent2id, rel2id, type2id;
map<unsigned, string> id2ent, id2rel, id2type;

//train data
long train_tri_num, train_pair_num;
vector<triple> train_tris;
vector<hepair> train_pairs;

//store data
map<unsigned, map<unsigned, set<unsigned> > > sub_rel_objs;
map<unsigned, vector<unsigned> > rel_heads, rel_tails;
vector<double> head_num_per_tail, tail_num_per_head;//平均每个head有多少个tail, 平均每个tail有多少个head
map<unsigned, set<unsigned> > ent_types;

/*************** variables and method using for test ***************/
double best_precision_bern = 0, best_precision_unif = 0;

double thre_entire_bern, thre_entire_unif;//全体阈值
map<unsigned, double> thre_rels_bern, thre_rels_unif;//每个关系的阈值

vector<triple> valid_pos_bern, valid_neg_bern, test_pos_bern, test_neg_bern;
vector<triple> valid_pos_unif, valid_neg_unif, test_pos_unif, test_neg_unif;

void eval_loadCorpus();
double evaluation(bool isbern);
/*************** variables and method using for test ***************/

bool exist(triple &tri){
	if (sub_rel_objs.count(tri.h) == 0)
		return false;
	if (sub_rel_objs[tri.h].count(tri.r) == 0)
		return false;
	if (sub_rel_objs[tri.h][tri.r].count(tri.t) == 0)
		return false;
	return true;
}

bool exist(hepair &pr){
	if (ent_types.count(pr.l) == 0)
		return false;
	if (ent_types[pr.l].count(pr.r) == 0)
		return false;
	return true;
}

void saveModel(int epoch){
	double precision = evaluation(Neg_Method);
	if (Neg_Method == 1){
		cout << "test precision(" << version << "): " << precision << endl;
		if (precision > best_precision_unif) best_precision_unif = precision;
		cout << "best precision(" << version << "): " << best_precision_unif << endl;
	}
	else{
		cout << "test precision(" << version << "): " << precision << endl;
		if (precision > best_precision_bern) best_precision_bern = precision;
		cout << "best precision(" << version << "): " << best_precision_bern << endl;
	}	

	string model_name = dim_str + "." + version;
	FILE* f1 = fopen(("entity2vec." + model_name).c_str(), "w");
	for (int ee = 0; ee < ent_num; ee++){
		for (int dd = 0; dd < n; dd++)
			fprintf(f1, "%.6lf\t", ent_vec[ee][dd]);
		fprintf(f1, "\n");
	}
	fclose(f1);
	FILE* f2 = fopen(("relation2vec." + model_name).c_str(), "w");
	for (int rr = 0; rr < rel_num; rr++){
		for (int dd = 0; dd < n; dd++)
			fprintf(f2, "%.6lf\t", rel_vec[rr][dd]);
		fprintf(f2, "\n");
	}
	fclose(f2);
	FILE* f3 = fopen(("type2vec." + model_name).c_str(), "w");
	for (int tt = 0; tt < type_num; tt++){
		for (int dd = 0; dd < n; dd++)
			fprintf(f3, "%.6lf\t", type_vec[tt][dd]);
		fprintf(f3, "\n");
	}
	fclose(f3);
}

void paramater_update(map<unsigned, vector<double> > &grad_temp, vector<vector<double> > &vec, bool need_normalize){
	for (map<unsigned, vector<double> >::iterator it_inner = grad_temp.begin();
		it_inner != grad_temp.end(); it_inner++){
		int id = it_inner->first;
		for (int ii = 0; ii < n; ii++){
			double grad = it_inner->second[ii];
			vec[id][ii] -= (rate * grad);
		}
		if (need_normalize)
			normalize(vec[id]);
	}
}

triple sampleNegTriple(unsigned pos_tri_id, bool is_head){
	bool in_relation = Neg_Scope;//是否在关系中选择
	triple tri_neg(train_tris[pos_tri_id]);
	int loop_size = 0;
	while (1){
		if (in_relation){
			if (is_head) tri_neg.h = rel_heads[tri_neg.r][rand() % rel_heads[tri_neg.r].size()];
			else tri_neg.t = rel_tails[tri_neg.r][rand() % rel_tails[tri_neg.r].size()];
		}
		else{
			if (is_head) tri_neg.h = rand() % ent_num;
			else tri_neg.t = rand() % ent_num;
		}
		if (!exist(tri_neg)) break;
		else if (loop_size++ > 10) in_relation = 0;//连续10次收不到，则在全局中抽
	}
	return tri_neg;
}

triple sampleNegTriple(unsigned pos_tri_id){
	int head_pro = 500;//选择调换head作为负样本的概率
	if (Neg_Method == 2){//bern
		double tph = tail_num_per_head[train_tris[pos_tri_id].r];
		double hpt = head_num_per_tail[train_tris[pos_tri_id].r];
		head_pro = 1000 * tph / (tph + hpt);
	}
	bool is_head = false;
	if ((rand() % 1000) < head_pro)
		is_head = true;
	return sampleNegTriple(pos_tri_id, is_head);
}

void trainTriple(triple &pos_tri, triple &neg_tri){
	vector<double> pos_hr = add(ent_vec_tmp[pos_tri.h], rel_vec_tmp[pos_tri.r]);
	vector<double> neg_hr = add(ent_vec_tmp[neg_tri.h], rel_vec_tmp[neg_tri.r]);
	vector<double> pos_diff = sub(pos_hr, ent_vec_tmp[pos_tri.t]);
	vector<double> neg_diff = sub(neg_hr, ent_vec_tmp[neg_tri.t]);

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
		paramater_update(entity_vec_grad_temp, ent_vec, true);
		paramater_update(relation_vec_grad_temp, rel_vec, true);
	}
}

void trainTriple(unsigned tri_id){	
	triple tri_neg = sampleNegTriple(tri_id);
	trainTriple(train_tris[tri_id], tri_neg);
}

void trainTriple(){
	vector<unsigned> batch_list(train_tri_num);
	for (int k = 0; k < train_tri_num; k++) batch_list[k] = k;
	random_disorder_list(batch_list);

	//random select batch，0 - triple_num
	if (Batch_Size == 0){
		ent_vec_tmp = ent_vec;
		rel_vec_tmp = rel_vec;
		#pragma omp parallel for schedule(dynamic) num_threads(THREAD_NUM)
		for (int index = 0; index < train_tri_num; index++)
			trainTriple(batch_list[index]);
		return;
	}

	int batchs = train_tri_num / Batch_Size;//每个batch有batch_size个样本

	for (int bat = 0; bat < batchs; bat++){
		int start = bat * Batch_Size;
		int end = (bat + 1) * Batch_Size;
		if (end > train_tri_num)
			end = train_tri_num;
		ent_vec_tmp = ent_vec;
		rel_vec_tmp = rel_vec;
		#pragma omp parallel for schedule(dynamic) num_threads(THREAD_NUM)
		for (int index = start; index < end; index++)
			trainTriple(batch_list[index]);
	}
}

void trainPair(hepair &pos_pair, hepair &neg_pair){
	vector<double> pos_diff = sub(ent_vec_tmp[pos_pair.l], type_vec_tmp[pos_pair.r]);
	vector<double> neg_diff = sub(ent_vec_tmp[neg_pair.l], type_vec_tmp[neg_pair.r]);

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
	map<unsigned, vector<double> > type_vec_grad_temp;
	entity_vec_grad_temp[pos_pair.l].resize(n);
	entity_vec_grad_temp[neg_pair.l].resize(n);
	type_vec_grad_temp[pos_pair.r].resize(n);
	type_vec_grad_temp[neg_pair.r].resize(n);

	//求解梯度
	double loss;
	for (int dd = 0; dd < n; dd++){
		//pos
		loss = pos_diff[dd];
		if (loss != 0){
			if (L1_Flag)
				if (loss > 0) loss = 1; else loss = -1;
			entity_vec_grad_temp[pos_pair.l][dd] += loss;
			type_vec_grad_temp[pos_pair.r][dd] -= loss;
		}
		//neg
		loss = neg_diff[dd];
		if (loss != 0){
			if (L1_Flag)
				if (loss > 0) loss = 1; else loss = -1;
			loss *= -1;
			entity_vec_grad_temp[neg_pair.l][dd] += loss;
			type_vec_grad_temp[neg_pair.r][dd] -= loss;
		}
	}

	#pragma omp critical
	{
		paramater_update(entity_vec_grad_temp, ent_vec, true);
		paramater_update(type_vec_grad_temp, type_vec, true);
	}
}

void trainPair(unsigned pair_id){
	hepair pos_pair = train_pairs[pair_id];	
	{//抽取负类
		hepair neg_type(pos_pair);
		while (1){
			neg_type.r = rand() % type_num;
			if (!exist(neg_type)) break;
		}
		trainPair(pos_pair, neg_type);
	}	
	{//抽取负实体
		hepair neg_ent(pos_pair);
		while (1){
			neg_ent.l = rand() % ent_num;
			if (!exist(neg_ent)) break;
		}
		trainPair(pos_pair, neg_ent);
	}
}

void trainPair(){
	vector<unsigned> batch_list(train_pair_num);
	for (int k = 0; k < train_pair_num; k++) batch_list[k] = k;
	random_disorder_list(batch_list);

	if (Batch_Size == 0){
		ent_vec_tmp = ent_vec;
		type_vec_tmp = type_vec;
		#pragma omp parallel for schedule(dynamic) num_threads(THREAD_NUM)
		for (int index = 0; index < train_pair_num; index++)
			trainPair(batch_list[index]);
		return;
	}	
	int batches = train_pair_num / Batch_Size;
	for (int bat = 0; bat < batches; bat++){
		int start = bat * Batch_Size;
		int end = (bat + 1) * Batch_Size;
		if (end > train_pair_num) end = train_pair_num;
		ent_vec_tmp = ent_vec;
		type_vec_tmp = type_vec;
		#pragma omp parallel for schedule(dynamic) num_threads(THREAD_NUM)
		for (int index = start; index < end; index++)
			trainPair(batch_list[index]);
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

		trainTriple();
		trainPair();

		lt = time(NULL);
		cout << "epoch " << epoch << " over  at: " << ctime(&lt);
		cout << "last loss sum : " << last_loss_sum << endl;
		cout << "this loss sum : " << loss_sum << endl;
		cout << "*************************" << endl;
		saveModel(epoch);
	}
}

void initModel(){
	if (Neg_Method == 1) version = "unif"; else version = "bern";
	char dim_ch[5]; sprintf(dim_ch, "%d", n); dim_str = dim_ch;

	ent_vec.resize(ent_num);
	for (int ee = 0; ee < ent_num; ee++){
		ent_vec[ee].resize(n);
		for (int dd = 0; dd < n; dd++)
			ent_vec[ee][dd] = rand(-1, 1);
		normalize(ent_vec[ee]);
	}
	rel_vec.resize(rel_num);
	for (int rr = 0; rr < rel_num; rr++){
		rel_vec[rr].resize(n);
		for (int dd = 0; dd < n; dd++)
			rel_vec[rr][dd] = rand(-1, 1);
		normalize(rel_vec[rr]);
	}
	type_vec.resize(type_num);
	for (int tt = 0; tt < type_num; tt++){
		type_vec[tt].resize(n);
		for (int dd = 0; dd < n; dd++)
			type_vec[tt][dd] = rand(-1, 1);
		normalize(type_vec[tt]);
	}

	//目前只有利用类别，转换矩阵默认为单位正	
	/*trans_mat.resize(1);

	for (int kk = 0; kk < 1; kk++){
		trans_mat[kk].resize(n);
		for (int rr = 0; rr < n; rr++){
			trans_mat[kk][rr].resize(n, 0);
			trans_mat[kk][rr][rr] = 1;
		}
	}*/
	cout << "init entity vectors, relation vectors and type vectors are over" << endl;
}

void prepare(){
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
	FILE *f_type_id = fopen((model_base_path + "../data/type2id.txt").c_str(), "r");
	while (fscanf(f_type_id, "%s%d", buf, &id) == 2){
		string type = buf; type2id[type] = id; id2type[id] = type; type_num++;
	}
	fclose(f_type_id);

	cout << "entity number = " << ent_num << endl;
	cout << "relation number = " << rel_num << endl;
	cout << "type number = " << type_num << endl;

	unsigned sub_id, rel_id, obj_id;
	string line;

	//读取三元组
	ifstream f_kb((model_base_path + "../data/train.txt").c_str());
	map<unsigned, set<unsigned> > rel_heads_temp, rel_tails_temp;

	map<unsigned, map<unsigned, set<unsigned> > > relation_head_tails;//计算平均一个head有多少个tail
	map<unsigned, map<unsigned, set<unsigned> > > relation_tail_heads;//计算平均一个tail有多少个head

	while (getline(f_kb, line)){
		vector<string> terms = split(line, "\t");
		sub_id = ent2id[terms[0]]; rel_id = rel2id[terms[1]]; obj_id = ent2id[terms[2]];
		train_tris.push_back(triple(sub_id, rel_id, obj_id)); train_tri_num++;
		sub_rel_objs[sub_id][rel_id].insert(obj_id);
		rel_heads_temp[rel_id].insert(sub_id);
		rel_tails_temp[rel_id].insert(obj_id);
		relation_head_tails[rel_id][sub_id].insert(obj_id);
		relation_tail_heads[rel_id][obj_id].insert(sub_id);
	}
	f_kb.close();
	cout << "tripe number = " << train_tri_num << endl;

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

	//读取类型
	unsigned ent_id, type_id;
	ifstream f_kb_type((model_base_path + "../data/train.txt").c_str());

	while (getline(f_kb_type, line)){
		vector<string> terms = split(line, "\t");
		ent_id = ent2id[terms[0]];
		for (int i = 1; i < terms.size(); i++){
			type_id = type2id[terms[i]];
			train_pairs.push_back(hepair(ent_id, type_id)); train_pair_num++;
			ent_types[ent_id].insert(type_id);
		}		
	}
	f_kb_type.close();
	cout << "entity type number = " << train_pair_num << endl;

}


double loss_triple(triple tri){
	vector<double> head = add(ent_vec[tri.h], rel_vec[tri.r]);
	vector<double> tail = ent_vec[tri.t];
	if (L1_Flag == 1)
		return l1_distance(head, tail);
	else
		return square_l2_distance(head, tail) / 2;
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
		triple tri(ent2id[subject], rel2id[relation], ent2id[object]);
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
	if ((i = ArgPos((char *)"-batch", argc, argv)) > 0) Batch_Size = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-epoch", argc, argv)) > 0) Epoch_Size = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) n = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-rate", argc, argv)) > 0) rate = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-margin", argc, argv)) > 0) margin = atof(argv[i + 1]);

	cout << "L1 / L2 = " << L1_Flag << endl;
	cout << "negative scope = " << Neg_Scope << endl;
	cout << "negative method = " << Neg_Method << endl;
	cout << "batch = " << Batch_Size << endl;
	cout << "epoch = " << Epoch_Size << endl;
	cout << "dim = " << n << endl;
	cout << "rate = " << rate << endl;
	cout << "margin = " << margin << endl;
	
	time_t lt = time(NULL);
	cout << "begin at: " << ctime(&lt);
	prepare();
	eval_loadCorpus();
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