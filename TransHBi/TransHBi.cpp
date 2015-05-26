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
#include <omp.h>
#include "experiment.h"

using namespace std;


#define E 2.718281828459
#define THREAD_NUM 64


double EPSILON = 1e-6;


bool L1_flag = 1;//包含L1/L2两个版本
int Grad_Method = 1;//1为SGD, 2为AdaGrad默认为1；今后加入其它梯度更新方法

int n = 50; //dimension of entity and relation
int batch_size = 120;
int epoch_size = 1000;
double rate = 0.001;//当使用AdaGrad的时候设置为1
double c_value = 0.0625;//实体归一化权重
double margin = 1;

int neg_scope = 0;//1表示限定关系，0表示全体
int method = 1; //1表示bern, 0表示unif

//global variables
vector<vector<double> > entity_vec, entity_vec_temp; //entity vector
vector<vector<double> > relation_vec, relation_vec_temp; //relation vector，一个关系对应正反两个向量

//for AdaGrad gradient update, sum of square of every steps
vector<vector<double> > entity_vec_grad_sum;
vector<vector<double> > relation_vec_grad_sum;

//origin data
long entity_num, relation_num;
map<string, unsigned> entity2id, relation2id;
map<unsigned, string> id2entity, id2relation;

//store the entire relation between entity and relations
long train_tris_num;
vector<triple> train_tris;

map<unsigned, map<unsigned, set<unsigned> > > subject_relation_objects;
map<unsigned, vector<unsigned> > relation_heads, relation_tails;//the head and tails with relation
vector<double> head_num_per_tail, tail_num_per_head;//平均每个head有多少个tail, 平均每个tail有多少个head


string version;
string dim_str;

double loss_sum = 0;

//string model_base_path = "G:/temp/TransX/wn11/TransH/";
string model_base_path = "";//currrent path

/*************** variables and method using for test ***************/
double best_precision_bern = 0, best_precision_unif = 0;

double thre_entire_bern, thre_entire_unif;//全体阈值
map<unsigned, double> thre_rels_bern, thre_rels_unif;//每个关系的阈值

vector<triple> valid_pos_bern, valid_neg_bern, test_pos_bern, test_neg_bern;
vector<triple> valid_pos_unif, valid_neg_unif, test_pos_unif, test_neg_unif;

void eval_prepare();
double evaluation(bool isbern);
/*************** variables and method using for test ***************/

void saveModel(int epoch){
	double precision = evaluation(method);
	cout << "-------------------------" << endl;
	cout << "test precision(" << version << "): " << precision << endl;
	cout << "best precision(" << version << "): ";
	if (method == 1){
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
	for (int kk = 0; kk < entity_num; kk++){
		for (int dim = 0; dim < n; dim++)
			fprintf(f1, "%.6lf\t", entity_vec[kk][dim]);
		fprintf(f1, "\n");
	}
	fclose(f1);
	FILE* f2 = fopen(("relation2vec." + dim_str + "." + version).c_str(), "w");
	for (int kk = 0; kk < relation_num * 2; kk++){
		for (int dim = 0; dim < n; dim++)
			fprintf(f2, "%.6lf\t", relation_vec[kk][dim]);
		fprintf(f2, "\n");
	}
	fclose(f2);
}

bool exist(unsigned sub, unsigned rel, unsigned obj){
	if (subject_relation_objects.count(sub) == 0)
		return false;
	if (subject_relation_objects[sub].count(rel) == 0)
		return false;
	if (subject_relation_objects[sub][rel].count(obj) == 0)
		return false;
	return true;
}



//update the paramaters, include the AdaGrad gradient and vector reprentation
void paramater_update(
	map<unsigned, vector<double> > &entity_vec_grad_temp,
	map<unsigned, vector<double> > &relation_vec_grad_temp){

	for (map<unsigned, vector<double> >::iterator it_inner = entity_vec_grad_temp.begin();
		it_inner != entity_vec_grad_temp.end(); it_inner++){

		int ent_id = it_inner->first;
		for (int ii = 0; ii < n; ii++){
			double grad = it_inner->second[ii];
			if (Grad_Method == 2){
				entity_vec_grad_sum[ent_id][ii] += square(grad);
				entity_vec[ent_id][ii] -= (grad * fast_rev_sqrt(entity_vec_grad_sum[ent_id][ii] + EPSILON) * rate);
			}
			else{
				entity_vec[ent_id][ii] -= (rate * grad);
			}
		}
	}

	for (map<unsigned, vector<double> >::iterator it_inner = relation_vec_grad_temp.begin();
		it_inner != relation_vec_grad_temp.end(); it_inner++){
		int rel_id = it_inner->first;
		for (int kk = 0; kk < n; kk++){
			double grad = it_inner->second[kk];
			if (Grad_Method == 2){
				relation_vec_grad_sum[rel_id][kk] += square(grad);
				relation_vec[rel_id][kk] -= (grad * fast_rev_sqrt(relation_vec_grad_sum[rel_id][kk] + EPSILON) * rate);
			}
			else{
				relation_vec[rel_id][kk] -= (rate * grad);
			}
		}
		normalize(relation_vec[rel_id]);//对关系向量进行归一化
	}
}

//对规整化实体部分求导
void gradientEntityOne(unsigned ent_id, map<unsigned, vector<double> > &entity_vec_grad_temp){
	if (norm_2(entity_vec_temp[ent_id]) > 1){
		for (int dd = 0; dd < n; dd++)
			entity_vec_grad_temp[ent_id][dd] += c_value * entity_vec_temp[ent_id][dd];
	}
}

//对规整化关系部分求导
void gradientRelation(unsigned rel_for, unsigned rel_rev, map<unsigned, vector<double> > &relation_vec_grad_temp){
	double inner_value = inner(relation_vec_temp[rel_for], relation_vec_temp[rel_rev]);
	for (int ii = 0; ii < n; ii++){
		for (int jj = 0; jj < n; jj++)
		{
			double pi = relation_vec_temp[rel_for][ii];
			double pj = relation_vec_temp[rel_for][jj];
			double qi = relation_vec_temp[rel_rev][ii];
			double qj = relation_vec_temp[rel_rev][jj];
			double pipj = pi * pj;
			double qiqj = qi * qj;
			double piqj = pi * qj;
			double ij_loss = pipj + qiqj - inner_value * piqj;
			if (fabs(ij_loss) <= EPSILON) continue;
			ij_loss *= c_value;
			double grad;
			//对pi求导
			grad = pj - inner_value * qj;
			grad -= piqj * qi;
			relation_vec_grad_temp[rel_for][ii] += grad * ij_loss;
			//对pj求导
			grad = pi - piqj * qj;
			relation_vec_grad_temp[rel_for][jj] += grad * ij_loss;
			//对qi求导
			grad = qj - piqj * pi;
			relation_vec_grad_temp[rel_rev][ii] += grad * ij_loss;
			//对qj求导
			grad = qi - inner_value * pi;
			grad -= piqj * pj;
			relation_vec_grad_temp[rel_rev][jj] += grad * ij_loss;
		}
	}
}


void gradientTriple(unsigned sub, unsigned rel, unsigned obj,
	vector<double> &ent_vec, vector<double> &ent_for,
	map<unsigned, vector<double> > &entity_vec_grad_temp,
	map<unsigned, vector<double> > &relation_vec_grad_temp, int sign)
{
	vector<double> pos_neg_diff(n, 0); //前部分减去后部分
	for (int dd = 0; dd < n; dd++){
		double tmp = ent_vec[dd] - ent_for[dd];
		if (L1_flag){
			if (tmp > 0) tmp = 1;
			else tmp = -1;
		}
		pos_neg_diff[dd] = sign * tmp;
	}

	/*HEAD部分，涉及sub, rel
	预先计算*/
	double d_r_h = 0, r_h = 0;
	for (int kk = 0; kk < n; kk++){
		d_r_h += pos_neg_diff[kk] * relation_vec_temp[rel][kk];
		r_h += relation_vec_temp[rel][kk] * entity_vec_temp[sub][kk];
	}
	for (int dd = 0; dd < n; dd++){
		//对sub求导
		entity_vec_grad_temp[sub][dd] += pos_neg_diff[dd];
		entity_vec_grad_temp[sub][dd] -= d_r_h * relation_vec_temp[rel][dd];
		//对pos的rel求导
		relation_vec_grad_temp[rel][dd] -= r_h * pos_neg_diff[dd];
		relation_vec_grad_temp[rel][dd] -= d_r_h * entity_vec_temp[sub][dd];
	}

	//TAIL部分，涉及obj, rel+relation_num
	unsigned rev_rel = rel + relation_num;
	double d_r_t = 0, r_t = 0;
	for (int kk = 0; kk < n; kk++){
		d_r_t += pos_neg_diff[kk] * relation_vec_temp[rev_rel][kk];
		r_t += relation_vec_temp[rev_rel][kk] * entity_vec_temp[obj][kk];
	}
	for (int dd = 0; dd < n; dd++){
		//对obj求导
		entity_vec_grad_temp[obj][dd] -= pos_neg_diff[dd];
		entity_vec_grad_temp[obj][dd] += d_r_t * relation_vec_temp[rev_rel][dd];
		//对pos的rel求导
		relation_vec_grad_temp[rev_rel][dd] += r_t * pos_neg_diff[dd];
		relation_vec_grad_temp[rev_rel][dd] += d_r_t * entity_vec_temp[obj][dd];
	}
}


//给定正负三元组，求相关实体和关系的梯度
int gradient(unsigned rel,
	unsigned pos_sub, unsigned pos_obj,
	unsigned neg_sub, unsigned neg_obj,
	map<unsigned, vector<double> > &entity_vec_grad_temp,
	map<unsigned, vector<double> > &relation_vec_grad_temp)
{
	vector<double> pos_head = transHyper(entity_vec_temp[pos_sub], relation_vec_temp[rel]);
	vector<double> pos_tail = transHyper(entity_vec_temp[pos_obj], relation_vec_temp[rel + relation_num]);

	vector<double> neg_head = transHyper(entity_vec_temp[neg_sub], relation_vec_temp[rel]);
	vector<double> neg_tail = transHyper(entity_vec_temp[neg_obj], relation_vec_temp[rel + relation_num]);


	double pos_loss, neg_loss;
	if (L1_flag){
		pos_loss = L1_distance(pos_head, pos_tail);
		neg_loss = L1_distance(neg_head, neg_tail);
	}
	else{
		pos_loss = square_L2_distance(pos_head, pos_tail) / 2;
		neg_loss = square_L2_distance(neg_head, neg_tail) / 2;
	}

	if (pos_loss + margin <= neg_loss)
		return 0;
	loss_sum += (pos_loss + margin - neg_loss);//该值越小越好

	//初始化所有需要求解的梯度
	entity_vec_grad_temp[pos_sub].resize(n, 0);
	entity_vec_grad_temp[pos_obj].resize(n, 0);
	if (neg_sub == pos_sub)
		entity_vec_grad_temp[neg_obj].resize(n, 0);
	else
		entity_vec_grad_temp[neg_sub].resize(n, 0);
	relation_vec_grad_temp[rel].resize(n, 0);
	relation_vec_grad_temp[rel + relation_num].resize(n, 0);

	//对pos三元组求导
	gradientTriple(pos_sub, rel, pos_obj, pos_head, pos_tail, entity_vec_grad_temp, relation_vec_grad_temp, 1);
	//对neg三元组求导
	gradientTriple(neg_sub, rel, neg_obj, neg_head, neg_tail, entity_vec_grad_temp, relation_vec_grad_temp, -1);

	//规整化部分
	gradientEntityOne(pos_sub, entity_vec_grad_temp);
	gradientEntityOne(pos_obj, entity_vec_grad_temp);
	if (neg_sub == pos_sub)
		gradientEntityOne(neg_obj, entity_vec_grad_temp);
	else
		gradientEntityOne(neg_sub, entity_vec_grad_temp);
	gradientRelation(rel, rel + relation_num, relation_vec_grad_temp);

	return 1;
}

void trainKB(unsigned rel,
	unsigned pos_sub, unsigned pos_obj,
	unsigned neg_sub, unsigned neg_obj){

	map<unsigned, vector<double> > entity_vec_grad_temp;
	map<unsigned, vector<double> > relation_vec_grad_temp;

	//compute gardient
	int grad_result = gradient(rel, pos_sub, pos_obj, neg_sub, neg_obj, entity_vec_grad_temp, relation_vec_grad_temp);
	if (grad_result == 0)
		return;

	#pragma omp critical
	{
		paramater_update(entity_vec_grad_temp, relation_vec_grad_temp);
	}
}

void trainKB(unsigned sub, unsigned rel, unsigned obj){
	int head_pro = 500;//选择调换head作为负样本的概率
	if (method){//bern
		double tph = tail_num_per_head[rel];
		double hpt = head_num_per_tail[rel];
		head_pro = 1000 * tph / (tph + hpt);
	}

	unsigned neg_sub = sub;
	unsigned neg_obj = obj;

	int count = 0;
	bool in_relation = neg_scope;//是否在关系中选择
	//随机从head中选择
	if ((rand() % 1000) < head_pro){
		int loop_size = 0;
		while (count < 1){
			if (in_relation)
				neg_sub = relation_heads[rel][rand() % relation_heads[rel].size()];
			else
				neg_sub = rand() % entity_num;
			if (exist(neg_sub, rel, obj)){
				if (loop_size++ > 10) in_relation = 0;
			}
			count++;
		}
	}
	else{
		int loop_size = 0;
		while (count < 1){
			if (in_relation)
				neg_obj = relation_tails[rel][rand() % relation_tails[rel].size()];
			else
				neg_obj = rand() % entity_num;
			if (exist(sub, rel, neg_obj)){
				if (loop_size++ > 10) in_relation = 0;
			}
			count++;
		}
	}
	trainKB(rel, sub, obj, neg_sub, neg_obj);
}

void trainModel(){
	time_t lt;

	for (int epoch = 0; epoch < epoch_size; epoch++){
		lt = time(NULL);
		cout << "*************************" << endl;
		cout << "epoch " << epoch << " begin at: " << ctime(&lt);
		double last_loss_sum = loss_sum;
		loss_sum = 0;

		//random select batch，0 - triple_num
		vector<unsigned> batch_list(train_tris_num);
		for (int k = 0; k < train_tris_num; k++)
			batch_list[k] = k;
		random_disorder_list(batch_list);

		int batchs = train_tris_num / batch_size;//每个batch有batch_size个样本

		for (int bat = 0; bat < batchs; bat++){
			int start = bat * batch_size;
			int end = (bat + 1) * batch_size;
			if (end > train_tris_num)
				end = train_tris_num;

			entity_vec_temp = entity_vec;
			relation_vec_temp = relation_vec;

			#pragma omp parallel for schedule(dynamic) num_threads(THREAD_NUM)
			for (int index = start; index < end; index++){
				int id = batch_list[index];
				trainKB(train_tris[id].h, train_tris[id].r, train_tris[id].t);
			}
				
		}

		lt = time(NULL);
		cout << "epoch " << epoch << " over  at: " << ctime(&lt);
		cout << "last loss sum : " << last_loss_sum << endl;
		cout << "this loss sum : " << loss_sum << endl;
		cout << "*************************" << endl;
		saveModel(epoch);
	}
}


/***************** 初始化神经网络 ***********************/
void initNet()
{
	entity_vec.resize(entity_num);
	for (unsigned kk = 0; kk < entity_vec.size(); kk++){
		entity_vec[kk].resize(n);
		for (int dd = 0; dd < n; dd++)
			entity_vec[kk][dd] = rand(-1, 1);
		normalize(entity_vec[kk]);
	}
	relation_vec.resize(relation_num * 2);
	for (int kk = 0; kk < relation_num * 2; kk++){
		relation_vec[kk].resize(n);
		for (int dd = 0; dd < n; dd++)
			relation_vec[kk][dd] = rand(-1, 1);
		normalize(relation_vec[kk]);
	}
	cout << "init entity vector and relation matrix is over." << endl;

	//or AdaGrad gradient update, sum of square of every steps
	entity_vec_grad_sum.resize(entity_num);
	for (int kk = 0; kk < entity_num; kk++)
		entity_vec_grad_sum[kk].resize(n, 0);
	relation_vec_grad_sum.resize(relation_num * 2);
	for (int kk = 0; kk < relation_num * 2; kk++){
		relation_vec_grad_sum[kk].resize(n, 0);
	}
	cout << "init entity adagrad vector and relation matrix adagrad is over." << endl;
}

void prepare(){
	char buf[1000];
	int id;
	FILE *f_ent_id = fopen((model_base_path + "../data/entity2id.txt").c_str(), "r");
	while (fscanf(f_ent_id, "%s%d", buf, &id) == 2){
		string ent = buf;
		entity2id[ent] = id;
		id2entity[id] = ent;
		entity_num++;
	}
	fclose(f_ent_id);

	FILE *f_rel_id = fopen((model_base_path + "../data/relation2id.txt").c_str(), "r");
	while (fscanf(f_rel_id, "%s%d", buf, &id) == 2){
		string rel = buf;
		relation2id[rel] = id;
		id2relation[id] = rel;
		relation_num++;
	}
	fclose(f_rel_id);
	cout << "entity number = " << entity_num << endl;
	cout << "relation number = " << relation_num << endl;

	FILE *f_kb = fopen((model_base_path + "../data/train.txt").c_str(), "r");
	string subject, relation, object;
	int subject_id, relation_id, object_id;

	map<unsigned, set<unsigned> > relation_heads_temp, relation_tails_temp;
	map<unsigned, map<unsigned, set<unsigned> > > relation_head_tails;//计算平均一个head有多少个tail
	map<unsigned, map<unsigned, set<unsigned> > > relation_tail_heads;//计算平均一个tail有多少个head

	while (fscanf(f_kb, "%s", buf) == 1){
		subject = buf;
		fscanf(f_kb, "%s", buf);
		relation = buf;
		fscanf(f_kb, "%s", buf);
		object = buf;

		if (entity2id.count(subject) == 0)
			cout << "miss entity:" << subject << endl;
		if (entity2id.count(object) == 0)
			cout << "miss entity:" << object << endl;
		if (relation2id.count(relation) == 0)
			cout << "miss relation:" << relation << endl;
		subject_id = entity2id[subject];
		relation_id = relation2id[relation];
		object_id = entity2id[object];
		train_tris_num++;
		train_tris.push_back(triple(subject_id, relation_id, object_id));

		subject_relation_objects[subject_id][relation_id].insert(object_id);

		relation_heads_temp[relation_id].insert(subject_id);
		relation_tails_temp[relation_id].insert(object_id);

		relation_head_tails[relation_id][subject_id].insert(object_id);
		relation_tail_heads[relation_id][object_id].insert(subject_id);
		
	}
	fclose(f_kb);
	cout << "triple number = " << train_tris_num << endl;

	for (map<unsigned, set<unsigned> >::iterator iter = relation_heads_temp.begin();
		iter != relation_heads_temp.end(); iter++){
		unsigned rel_id = iter->first;
		for (set<unsigned>::iterator inner_iter = iter->second.begin(); inner_iter != iter->second.end(); inner_iter++)
			relation_heads[rel_id].push_back(*inner_iter);
	}
	for (map<unsigned, set<unsigned> >::iterator iter = relation_tails_temp.begin();
		iter != relation_tails_temp.end(); iter++){
		unsigned rel_id = iter->first;
		for (set<unsigned>::iterator inner_iter = iter->second.begin(); inner_iter != iter->second.end(); inner_iter++)
			relation_tails[rel_id].push_back(*inner_iter);
	}

	tail_num_per_head.resize(relation_num);
	head_num_per_tail.resize(relation_num);
	for (int rel_id = 0; rel_id < relation_num; rel_id++){
		//计算平均一个head有多少个tail
		map<unsigned, set<unsigned> > tails_per_head = relation_head_tails[rel_id];
		unsigned head_number = 0;
		unsigned tail_count = 0;
		for (map<unsigned, set<unsigned> > ::iterator iter = tails_per_head.begin();
			iter != tails_per_head.end(); iter++){
			if (iter->second.size() > 0){
				head_number++;
				tail_count += iter->second.size();
			}
		}
		tail_num_per_head[rel_id] = 1.0 * tail_count / head_number;
		//计算平均一个tail有多少个head
		map<unsigned, set<unsigned> > heads_per_tail = relation_tail_heads[rel_id];
		unsigned tail_number = 0;
		unsigned head_count = 0;
		for (map<unsigned, set<unsigned> > ::iterator iter = heads_per_tail.begin();
			iter != heads_per_tail.end(); iter++){
			if (iter->second.size() > 0){
				tail_number++;
				head_count += iter->second.size();
			}
		}
		head_num_per_tail[rel_id] = 1.0 * head_count / tail_number;
		//cout << id2relation[rel_id] << ". hpt=[" << head_num_per_tail[rel_id] << "], tph=[" << tail_num_per_head[rel_id] << "]" << endl;
	}

	eval_prepare();
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


void eval_prepare(){
	load_eval_data(true, true, true, valid_pos_bern);
	load_eval_data(true, true, false, valid_pos_unif);
	load_eval_data(true, false, true, valid_neg_bern);
	load_eval_data(true, false, false, valid_neg_unif);

	load_eval_data(false, true, true, test_pos_bern);
	load_eval_data(false, true, false, test_pos_unif);
	load_eval_data(false, false, true, test_neg_bern);
	load_eval_data(false, false, false, test_neg_unif);
}


double loss_triple(triple tri){
	unsigned sub = tri.h, rel = tri.r, obj = tri.t;
	vector<double> sub_vec = transHyper(entity_vec[sub], relation_vec[rel]);
	vector<double> obj_vec = transHyper(entity_vec[obj], relation_vec[rel + relation_num]);
	if (L1_flag == 1)
		return L1_distance(sub_vec, obj_vec);
	else
		return L2_distance(sub_vec, obj_vec);
}


void eval_valid(bool isbern){
	if (isbern){
		thre_entire_bern = 0;
		thre_rels_bern.clear();
		eval_valid(
			valid_pos_bern,
			valid_neg_bern,
			thre_rels_bern, thre_entire_bern, loss_triple);
	}
	else{
		thre_entire_unif = 0;
		thre_rels_unif.clear();
		eval_valid(
			valid_pos_unif,
			valid_neg_unif,
			thre_rels_unif, thre_entire_unif, loss_triple);
	}
}

double eval_test(bool isbern){
	if (isbern)
		return eval_test(
		test_pos_bern,
		test_neg_bern,
		thre_rels_bern, thre_entire_bern, loss_triple);
	else
		return eval_test(
		test_pos_unif,
		test_neg_unif,
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
	if ((i = ArgPos((char *)"-l1", argc, argv)) > 0) L1_flag = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-grad", argc, argv)) > 0) Grad_Method = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) n = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-batch", argc, argv)) > 0) batch_size = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-epoch", argc, argv)) > 0) epoch_size = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-rate", argc, argv)) > 0) rate = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-c", argc, argv)) > 0) c_value = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-margin", argc, argv)) > 0) margin = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-method", argc, argv)) > 0) method = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-negScope", argc, argv)) > 0) neg_scope = atoi(argv[i + 1]);


	if (method == 1)
		version = "bern";
	else
		version = "unif";
	char dim_ch[5];
	sprintf(dim_ch, "%d", n);
	dim_str = dim_ch;

	cout << "dim of entity = " << n << endl;
	cout << "batch num = " << batch_size << endl;
	cout << "epoch num = " << epoch_size << endl;
	cout << "learing rate = " << rate << endl;
	cout << "entity normalize weight(C) = " << c_value << endl;
	cout << "margin = " << margin << endl;
	cout << "method = " << version << endl;
	if (L1_flag) cout << "use L1 as dissimilarity" << endl;
	else cout << "use L2 as dissimilarity" << endl;
	if (Grad_Method == 2) cout << "use adaGrad optimization" << endl;
	else cout << "use SGD optimization" << endl;
	if (neg_scope) cout << "sample negative in relation scope" << endl;
	else cout << "sample negative in entire scope" << endl;

	time_t lt = time(NULL);
	cout << "begin at: " << ctime(&lt);
	prepare();

	lt = time(NULL);
	cout << "prepare over at: " << ctime(&lt);
	initNet();

	lt = time(NULL);
	cout << "init net over at: " << ctime(&lt);

	trainModel();
	lt = time(NULL);
	cout << "train over at: " << ctime(&lt);

	return 1;
}