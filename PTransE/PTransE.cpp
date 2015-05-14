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
#include "experiment.h"

using namespace std;

#define THREAD_NUM 4//线程个数
double EPSILON = 1e-6;

//string model_base_path = "G:/temp/TransX/fb13/JLeRR_2/";
string model_base_path = "";//currrent path
double loss_sum;

//global parameters
int  Grad_Method = 1;
int  Batch_Size = 480;
int  Epoch_Size = 1000;
int  Neg_Size = 15;
int    n = 50;
double rate = 0.001;

//global variables
vector<vector<double> > ent_vec, ent_vec_temp;
vector<vector<double> > rel_vec, rel_vec_temp;

//for AdaGrad gradient update
vector<vector<double> > ada_ent_vec, ada_rel_vec;

//origin data
long ent_num, rel_num;
map<string, unsigned> ent2id, rel2id;
map<unsigned, string> id2ent, id2rel;
long tri_num;
vector<triple> triples;

map<unsigned, map<unsigned, set<unsigned> > > sub_rel_objs, obj_rel_subs, sub_obj_rels;

/*************** variables and method using for test ***************/
double best_precision;

double thre_entire;//全体阈值
map<unsigned, double> thre_rels;//每个关系的阈值

vector<triple> valid_pos, valid_neg, test_pos, test_neg;

void eval_loadCorpus();
double evaluation();
/*************** variables and method using for test ***************/

void saveModel(int epoch){
	double precision = evaluation();
	cout << "-------------------------" << endl;
	cout << "test precision: " << precision << endl;
	cout << "best precision: " << best_precision << endl;	
	if (precision > best_precision)
		best_precision = precision;
	//else return;

	char dim_ch[5];
	sprintf(dim_ch, "%d", n);
	string dim_str = dim_ch;

	FILE* f1 = fopen(("entity2vec." + dim_str).c_str(), "w");
	for (int kk = 0; kk < ent_num; kk++){
		for (int dim = 0; dim < n; dim++)
			fprintf(f1, "%.6lf\t", ent_vec[kk][dim]);
		fprintf(f1, "\n");
	}
	fclose(f1);
	FILE* f2 = fopen(("relation2vec." + dim_str).c_str(), "w");
	for (int kk = 0; kk < rel_num; kk++){
		for (int dim = 0; dim < n; dim++)
			fprintf(f2, "%.6lf\t", rel_vec[kk][dim]);
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
				ada_ent_vec[ent_id][ii] += square(grad);
				ent_vec[ent_id][ii] -= (grad * fast_rev_sqrt(ada_ent_vec[ent_id][ii] + EPSILON) * rate);
			}
			else
				ent_vec[ent_id][ii] -= (rate * grad);
		}
		normalize(ent_vec[ent_id]);
	}

	for (map<unsigned, vector<double> >::iterator it_inner = relation_vec_grad_temp.begin();
		it_inner != relation_vec_grad_temp.end(); it_inner++){
		int rel_id = it_inner->first;
		for (int kk = 0; kk < n; kk++){
			double grad = it_inner->second[kk];
			if (Grad_Method == 2){
				ada_rel_vec[rel_id][kk] += square(grad);
				rel_vec[rel_id][kk] -= (grad * fast_rev_sqrt(ada_rel_vec[rel_id][kk] + EPSILON) * rate);
			}
			else
				rel_vec[rel_id][kk] -= (rate * grad);
		}
		normalize(rel_vec[rel_id]);
	}
}

double triple_sigmod(triple tri, bool is_pos){
	vector<double> ent = sub(ent_vec[tri.h], ent_vec[tri.t]);
	if (is_pos)
		return sigmod(inner(ent, rel_vec[tri.r]));
	else
		return sigmod(-1 * inner(ent, rel_vec[tri.r]));
}

void trainTriple(triple pos_tri, vector<triple> neg_tris, double weight){	
	double ratio = rate * weight;
	double sm_pos_tri = triple_sigmod(pos_tri, true);//正例sigmod值
	vector<double> sm_neg_tris(neg_tris.size());//所有负例sigmod值
	double loss_temp = log(sm_pos_tri);

	map<unsigned, vector<double> > entity_vec_grad_temp;
	map<unsigned, vector<double> > relation_vec_grad_temp;
	entity_vec_grad_temp[pos_tri.h].resize(n);
	entity_vec_grad_temp[pos_tri.t].resize(n);
	relation_vec_grad_temp[pos_tri.r].resize(n);
	for (int i = 0; i < neg_tris.size(); i++){
		entity_vec_grad_temp[neg_tris[i].h].resize(n);
		entity_vec_grad_temp[neg_tris[i].t].resize(n);
		relation_vec_grad_temp[neg_tris[i].r].resize(n);
		double sm_neg = triple_sigmod(neg_tris[i], false);
		sm_neg_tris.push_back(sm_neg);
		loss_temp += log(sm_neg);
	}

	loss_sum += loss_temp * weight;

	//对正例求导
	double temp_grad = ratio * (1 - sm_pos_tri);
	for (int dd = 0; dd < n; dd++){		
		entity_vec_grad_temp[pos_tri.h][dd] += temp_grad * rel_vec_temp[pos_tri.r][dd];
		entity_vec_grad_temp[pos_tri.t][dd] -= temp_grad * rel_vec_temp[pos_tri.r][dd];
		relation_vec_grad_temp[pos_tri.r][dd] += temp_grad * 
			(ent_vec_temp[pos_tri.h][dd] - ent_vec_temp[pos_tri.t][dd]);
	}
	//对负例求导
	for (int i = 0; i < neg_tris.size(); i++){
		temp_grad = ratio * (sm_neg_tris[i] - 1);
		for (int dd = 0; dd < n; dd++){
			entity_vec_grad_temp[neg_tris[i].h][dd] += temp_grad * rel_vec_temp[neg_tris[i].r][dd];
			entity_vec_grad_temp[neg_tris[i].t][dd] -= temp_grad * rel_vec_temp[neg_tris[i].r][dd];
			relation_vec_grad_temp[neg_tris[i].r][dd] += temp_grad * 
				(ent_vec_temp[neg_tris[i].h][dd] - ent_vec_temp[neg_tris[i].t][dd]);
		}
	}

	#pragma omp critical
	{
		paramater_update(entity_vec_grad_temp, relation_vec_grad_temp);
	}
}


void trainTriple(triple pos_tri){
	//h, r, t分别抽取负例
	double prop;
	int count = 0;
	set<unsigned> temp;
	//替换t
	vector<triple> tail_neg_tris;	
	while (count++ < Neg_Size){
		int neg_t = rand() % ent_num;
		//if (sub_rel_objs[pos_tri.h][pos_tri.r].count(neg_t) == 0 && temp.count(neg_t) == 0){
		if (temp.count(neg_t) == 0){
			triple neg_tri(pos_tri);
			neg_tri.t = neg_t;
			tail_neg_tris.push_back(neg_tri);
			temp.insert(neg_t);
		}
	}
	prop = 1.0 / sub_rel_objs[pos_tri.h][pos_tri.r].size();
	trainTriple(pos_tri, tail_neg_tris, prop);

	//替换r	
	count = 0;
	temp.clear();
	vector<triple> rel_neg_tris;
	while (count++ < Neg_Size){
		int neg_r = rand() % rel_num;
		//if (sub_obj_rels[pos_tri.h][pos_tri.t].count(neg_r) == 0 && temp.count(neg_r) == 0){
		if (temp.count(neg_r) == 0){
			triple neg_tri(pos_tri);
			neg_tri.r = neg_r;
			rel_neg_tris.push_back(neg_tri);
			temp.insert(neg_r);
		}
	}
	prop = 1.0 / sub_obj_rels[pos_tri.h][pos_tri.t].size();
	trainTriple(pos_tri, rel_neg_tris, prop);

	//替换h
	count = 0;
	temp.clear();
	vector<triple> head_neg_tris;
	while (count++ < Neg_Size){
		int neg_h = rand() % ent_num;
		//if (obj_rel_subs[pos_tri.t][pos_tri.r].count(neg_h) == 0 && temp.count(neg_h) == 0){
		if (temp.count(neg_h) == 0){
			triple neg_tri(pos_tri);
			neg_tri.h = neg_h;
			head_neg_tris.push_back(neg_tri);
			temp.insert(neg_h);
		}
	}
	prop = 1.0 / obj_rel_subs[pos_tri.t][pos_tri.r].size();
	trainTriple(pos_tri, head_neg_tris, prop);

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
		#pragma omp parallel for schedule(dynamic) num_threads(THREAD_NUM)
		for (int index = start; index < end; index++)
			trainTriple(triples[batch_list[index]]);
	}
}

//double energy_function(triple tri){
//	vector<double> head = add(ent_vec_temp[tri.h], rel_vec_temp[tri.r]);
//	if (L1_Flag)
//		return L1_distance(head, ent_vec_temp[tri.t]);
//	else
//		return square_L2_distance(head, ent_vec_temp[tri.t]) / 2;
//}

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
	cout << "init entity vector, relation vector and formula weights are over" << endl;

	//or AdaGrad gradient update, sum of square of every steps
	ada_ent_vec.resize(ent_num);
	for (int kk = 0; kk < ent_num; kk++)
		ada_ent_vec[kk].resize(n, 0);
	ada_rel_vec.resize(rel_num);
	for (int kk = 0; kk < rel_num; kk++)
		ada_rel_vec[kk].resize(n, 0);
	cout << "init adagrad parameters are over" << endl;
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
	
	while (getline(f_kb, line)){
		vector<string> terms = split(line, "\t");
		sub_id = ent2id[terms[0]]; rel_id = rel2id[terms[1]]; obj_id = ent2id[terms[2]];
		triples.push_back(triple(sub_id, rel_id, obj_id)); tri_num++;

		sub_rel_objs[sub_id][rel_id].insert(obj_id);
		sub_obj_rels[sub_id][obj_id].insert(rel_id);
		obj_rel_subs[obj_id][rel_id].insert(sub_id);
	}
	f_kb.close();
	cout << "tripe number = " << tri_num << endl;

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
		triple tri(ent2id[subject], rel2id[relation], ent2id[object]);
		triple_list.push_back(tri);
	}
	fclose(f_kb);
	cout << "load: " << filename << endl;
}


void eval_loadCorpus(){
	load_eval_data(true, true, true, valid_pos);
	load_eval_data(true, false, true, valid_neg);
	load_eval_data(false, true, true, test_pos);
	load_eval_data(false, false, true, test_neg);
}

double loss_triple(triple tri){
	vector<double> ent = sub(ent_vec[tri.h], ent_vec[tri.t]);
	return sigmod(inner(ent, rel_vec[tri.r]));
}


void eval_valid(){
	thre_entire = 0;
	thre_rels.clear();
	eval_valid(valid_pos, valid_neg, thre_rels, thre_entire, loss_triple);
}

double eval_test(){
	return eval_test(test_pos, test_neg, thre_rels, thre_entire, loss_triple);
}

double evaluation(){
	eval_valid();
	double test_score = eval_test();
	return test_score;
}
/********** end of triple classification **********/


int main(int argc, char**argv){
	int i;

	if ((i = ArgPos((char *)"-grad", argc, argv)) > 0) Grad_Method = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-batch", argc, argv)) > 0) Batch_Size = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-epoch", argc, argv)) > 0) Epoch_Size = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) n = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-rate", argc, argv)) > 0) rate = atof(argv[i + 1]);

	cout << "grad method = " << Grad_Method << endl;
	cout << "batch = " << Batch_Size << endl;
	cout << "epoch = " << Epoch_Size << endl;
	cout << "dim = " << n << endl;
	cout << "rate = " << rate << endl;

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




