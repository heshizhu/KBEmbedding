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
#define THREAD_NUM 4

double EPSILON = 1e-6;


bool L1_Flag = 1;//包含L1/L2两个版本
int Grad_Method = 1;//1为SGD, 2为AdaGrad默认为1；今后加入其它梯度更新方法
int Batch_Size = 120;
int Epoch_Size = 1000;
int Neg_Scope = 0;//1表示限定关系，0表示全体
int Neg_Method = 1; //1表示bern, 0表示unif
double rate = 0.001;//当使用AdaGrad的时候设置为1
double c_value = 0.0625;//实体归一化权重
double margin = 1;
int n = 50; //dimension of entity and relation

//global variables
vector<vector<double> > ent_vec, ent_vec_tmp; //entity vector
vector<vector<double> > rel_vec, rel_vec_tmp; //relation vector，一个关系对应正反两个向量

//for AdaGrad gradient update, sum of square of every steps
vector<vector<double> > ent_vec_grad_sum;
vector<vector<double> > rel_vec_grad_sum;

//origin data
long ent_num, rel_num;
map<string, unsigned> ent2id, rel2id;
map<unsigned, string> id2ent, id2rel;

//store the entire relation between entity and relations
long train_tris_num;
vector<triple> train_tris;

map<unsigned, map<unsigned, set<unsigned> > > sub_rel_objs;
map<unsigned, vector<unsigned> > rel_heads, rel_tails;//the head and tails with relation
vector<double> head_num_per_tail, tail_num_per_head;//平均每个head有多少个tail, 平均每个tail有多少个head

string version;
string dim_str;
double loss_sum = 0;
//string model_base_path = "G:/temp/TransX/wn11/TransH/";
string model_base_path = "";//currrent path


/*************** variables and method using for test ***************/
double best_precision = 0;
double thre_entire;//全体阈值
map<unsigned, double> thre_rels;//每个关系的阈值
vector<triple> valid_pos, valid_neg, test_pos, test_neg;
void eval_prepare();
double evaluation();
/*************** variables and method using for test ***************/

void saveModel(int epoch){
	double precision = evaluation();
	cout << "-------------------------" << endl;
	cout << "test precision(" << version << "): " << precision << endl;
	cout << "best precision(" << version << "): ";
	cout << best_precision << endl;
	if (precision > best_precision)
		best_precision = precision;
	else return;

	FILE* f1 = fopen(("entity2vec." + dim_str + "." + version).c_str(), "w");
	for (int kk = 0; kk < ent_num; kk++){
		for (int dim = 0; dim < n; dim++)
			fprintf(f1, "%.6lf\t", ent_vec[kk][dim]);
		fprintf(f1, "\n");
	}
	fclose(f1);
	FILE* f2 = fopen(("relation2vec." + dim_str + "." + version).c_str(), "w");
	for (int kk = 0; kk < rel_num * 3; kk++){
		for (int dim = 0; dim < n; dim++)
			fprintf(f2, "%.6lf\t", rel_vec[kk][dim]);
		fprintf(f2, "\n");
	}
	fclose(f2);
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
	map<unsigned, vector<double> > &entity_vec_grad_temp,
	map<unsigned, vector<double> > &relation_vec_grad_temp){

	for (map<unsigned, vector<double> >::iterator it_inner = entity_vec_grad_temp.begin();
		it_inner != entity_vec_grad_temp.end(); it_inner++){

		int ent_id = it_inner->first;
		for (int ii = 0; ii < n; ii++){
			double grad = it_inner->second[ii];
			if (Grad_Method == 2){
				ent_vec_grad_sum[ent_id][ii] += square(grad);
				ent_vec[ent_id][ii] -= (grad * fast_rev_sqrt(ent_vec_grad_sum[ent_id][ii] + EPSILON) * rate);
			}
			else{
				ent_vec[ent_id][ii] -= (rate * grad);
			}
		}
	}

	for (map<unsigned, vector<double> >::iterator it_inner = relation_vec_grad_temp.begin();
		it_inner != relation_vec_grad_temp.end(); it_inner++){
		int rel_id = it_inner->first;
		for (int kk = 0; kk < n; kk++){
			double grad = it_inner->second[kk];
			if (Grad_Method == 2){
				rel_vec_grad_sum[rel_id][kk] += square(grad);
				rel_vec[rel_id][kk] -= (grad * fast_rev_sqrt(rel_vec_grad_sum[rel_id][kk] + EPSILON) * rate);
			}
			else{
				rel_vec[rel_id][kk] -= (rate * grad);
			}
		}
		if (rel_id % 3 != 0)
			normalize2one(rel_vec[rel_id]);//对法向量归一化
		//normalize(rel_vec[rel_id]);//对关系向量进行归一化
	}
}

//对规整化实体部分求导
void gradientEntityOne(unsigned ent_id, map<unsigned, vector<double> > &entity_vec_grad_temp){
	if (norm_2(ent_vec_tmp[ent_id]) > 1){
		for (int dd = 0; dd < n; dd++)
			entity_vec_grad_temp[ent_id][dd] += c_value * ent_vec_tmp[ent_id][dd];
	}
}

//对规整化关系部分求导
void gradientRelation(unsigned rel_for, unsigned rel_rev, map<unsigned, vector<double> > &relation_vec_grad_temp){
	double inner_value = inner(rel_vec_tmp[rel_for], rel_vec_tmp[rel_rev]);
	for (int ii = 0; ii < n; ii++){
		for (int jj = 0; jj < n; jj++)
		{
			double pi = rel_vec_tmp[rel_for][ii];
			double pj = rel_vec_tmp[rel_for][jj];
			double qi = rel_vec_tmp[rel_rev][ii];
			double qj = rel_vec_tmp[rel_rev][jj];
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

//让法向量与关系向量正交
void gradientOrthodox(unsigned rel_id, unsigned rel_ort, map<unsigned, vector<double> > &rel_vec_grad_tmp){
	//r0为关系,r1为法向量
	double head_inner = inner(rel_vec_tmp[rel_ort], rel_vec_tmp[rel_id]);
	double tail_norm2 = norm_2(rel_vec_tmp[rel_id]);

	if (square(head_inner) <= EPSILON * square(tail_norm2))
		return;

	double temp1 = 2 * head_inner;
	double temp2 = 4 * EPSILON * tail_norm2;
	for (int dd = 0; dd < n; dd++){
		//r1部分
		rel_vec_grad_tmp[rel_ort][dd] += temp1 * rel_vec_tmp[rel_id][dd];
		//ro部分
		rel_vec_grad_tmp[rel_id][dd] += temp1 * rel_vec_tmp[rel_ort][dd];
		rel_vec_grad_tmp[rel_id][dd] -= temp2 * rel_vec_tmp[rel_id][dd];
	}
}

void gradientTriple(unsigned rel, unsigned rel_dir, unsigned rel_rev, unsigned sub, unsigned obj,
	vector<double> &ent_vec, vector<double> &ent_for,
	map<unsigned, vector<double> > &entity_vec_grad_temp,
	map<unsigned, vector<double> > &relation_vec_grad_temp, int sign)
{
	vector<double> pos_neg_diff(n, 0); //前部分减去后部分
	for (int dd = 0; dd < n; dd++){
		double tmp = ent_vec[dd] - ent_for[dd];
		if (L1_Flag){
			if (tmp > 0) tmp = 1;
			else tmp = -1;
		}
		pos_neg_diff[dd] = sign * tmp;
	}

	//关系部分，涉及rel
	for (int dd = 0; dd < n; dd++)
		relation_vec_grad_temp[rel][dd] += pos_neg_diff[dd];

	/*HEAD部分，涉及sub, rel_dir
	预先计算*/
	double d_r_h = 0, r_h = 0;
	for (int kk = 0; kk < n; kk++){
		d_r_h += pos_neg_diff[kk] * rel_vec_tmp[rel_dir][kk];
		r_h += rel_vec_tmp[rel_dir][kk] * ent_vec_tmp[sub][kk];
	}
	for (int dd = 0; dd < n; dd++){
		//对sub求导
		entity_vec_grad_temp[sub][dd] += pos_neg_diff[dd];
		entity_vec_grad_temp[sub][dd] -= d_r_h * rel_vec_tmp[rel_dir][dd];
		//对pos的rel求导
		relation_vec_grad_temp[rel_dir][dd] -= r_h * pos_neg_diff[dd];
		relation_vec_grad_temp[rel_dir][dd] -= d_r_h * ent_vec_tmp[sub][dd];
	}

	//TAIL部分，涉及obj, rel_rev
	double d_r_t = 0, r_t = 0;
	for (int kk = 0; kk < n; kk++){
		d_r_t += pos_neg_diff[kk] * rel_vec_tmp[rel_rev][kk];
		r_t += rel_vec_tmp[rel_rev][kk] * ent_vec_tmp[obj][kk];
	}
	for (int dd = 0; dd < n; dd++){
		//对obj求导
		entity_vec_grad_temp[obj][dd] -= pos_neg_diff[dd];
		entity_vec_grad_temp[obj][dd] += d_r_t * rel_vec_tmp[rel_rev][dd];
		//对pos的rel求导
		relation_vec_grad_temp[rel_rev][dd] += r_t * pos_neg_diff[dd];
		relation_vec_grad_temp[rel_rev][dd] += d_r_t * ent_vec_tmp[obj][dd];
	}
}


//给定正负三元组，求相关实体和关系的梯度
int gradient(triple &tri_pos, triple &tri_neg,
	map<unsigned, vector<double> > &ent_vec_grad_tmp,
	map<unsigned, vector<double> > &rel_vec_grad_tmp)
{
	unsigned rel = tri_pos.r, rel_dir = tri_pos.r + 1, rel_rev = tri_pos.r + 2;
	unsigned pos_sub = tri_pos.h, pos_obj = tri_pos.t;
	unsigned neg_sub = tri_neg.h, neg_obj = tri_neg.t;

	vector<double> pos_sub_vec = transHyper(ent_vec_tmp[pos_sub], rel_vec_tmp[rel_dir]);
	vector<double> pos_head = add(rel_vec_tmp[rel], pos_sub_vec);
	vector<double> pos_tail = transHyper(ent_vec_tmp[pos_obj], rel_vec_tmp[rel_rev]);

	vector<double> neg_sub_vec = transHyper(ent_vec_tmp[neg_sub], rel_vec_tmp[rel_dir]);
	vector<double> neg_head = add(rel_vec_tmp[rel], neg_sub_vec);
	vector<double> neg_tail = transHyper(ent_vec_tmp[neg_obj], rel_vec_tmp[rel_rev]);


	double pos_loss, neg_loss;
	if (L1_Flag){
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
	ent_vec_grad_tmp[pos_sub].resize(n, 0);
	ent_vec_grad_tmp[pos_obj].resize(n, 0);	
	ent_vec_grad_tmp[neg_sub].resize(n, 0);
	ent_vec_grad_tmp[neg_obj].resize(n, 0);	
	rel_vec_grad_tmp[rel].resize(n, 0);
	rel_vec_grad_tmp[rel_dir].resize(n, 0);
	rel_vec_grad_tmp[rel_rev].resize(n, 0);
	
	//对pos三元组求导
	gradientTriple(rel, rel_dir, rel_rev, pos_sub, pos_obj, pos_head, pos_tail, ent_vec_grad_tmp, rel_vec_grad_tmp, 1);
	//对neg三元组求导
	gradientTriple(rel, rel_dir, rel_rev, neg_sub, neg_obj, neg_head, neg_tail, ent_vec_grad_tmp, rel_vec_grad_tmp, -1);
	
	//规整化部分
	gradientEntityOne(pos_sub, ent_vec_grad_tmp);
	gradientEntityOne(pos_obj, ent_vec_grad_tmp);
	if (neg_sub == pos_sub)
		gradientEntityOne(neg_obj, ent_vec_grad_tmp);
	else
		gradientEntityOne(neg_sub, ent_vec_grad_tmp);
	gradientRelation(rel_dir, rel_rev, rel_vec_grad_tmp);
	gradientOrthodox(rel, rel_dir, rel_vec_grad_tmp);
	gradientOrthodox(rel, rel_rev, rel_vec_grad_tmp);
	
	return 1;
}

void trainKB(triple &tri_pos, triple &tri_neg){
	map<unsigned, vector<double> > ent_vec_grad_tmp;
	map<unsigned, vector<double> > rel_vec_grad_tmp;
	//compute gardient
	int grad_result = gradient(tri_pos, tri_neg, ent_vec_grad_tmp, rel_vec_grad_tmp);
	if (grad_result == 0)
		return;
	#pragma omp critical
	{
		paramater_update(ent_vec_grad_tmp, rel_vec_grad_tmp);		
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
	//1:unif; 2:bern
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

void trainKB(unsigned tri_id){
	triple tri_pos = train_tris[tri_id];
	triple tri_neg = sampleNegTriple(tri_id);
	trainKB(tri_pos, tri_neg);
}

void trainModel(){
	time_t lt;

	for (int epoch = 0; epoch < Epoch_Size; epoch++){
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

		int batchs = train_tris_num / Batch_Size;//每个batch有batch_size个样本

		for (int bat = 0; bat < batchs; bat++){
			int start = bat * Batch_Size;
			int end = (bat + 1) * Batch_Size;
			if (end > train_tris_num)
				end = train_tris_num;

			ent_vec_tmp = ent_vec;
			rel_vec_tmp = rel_vec;

			#pragma omp parallel for schedule(dynamic) num_threads(THREAD_NUM)
			for (int index = start; index < end; index++)
				trainKB(batch_list[index]);

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
	ent_vec.resize(ent_num);
	for (unsigned kk = 0; kk < ent_num; kk++){
		ent_vec[kk].resize(n);
		for (int dd = 0; dd < n; dd++)
			ent_vec[kk][dd] = rand(-1, 1);
		normalize(ent_vec[kk]);
	}
	rel_vec.resize(rel_num * 3);
	for (int kk = 0; kk < rel_num * 3; kk++){
		rel_vec[kk].resize(n);
		for (int dd = 0; dd < n; dd++)
			rel_vec[kk][dd] = rand(-1, 1);
		normalize(rel_vec[kk]);
	}
	cout << "init entity vector and relation matrix is over." << endl;

	//or AdaGrad gradient update, sum of square of every steps
	ent_vec_grad_sum.resize(ent_num);
	for (int kk = 0; kk < ent_num; kk++)
		ent_vec_grad_sum[kk].resize(n, 0);
	rel_vec_grad_sum.resize(rel_num * 3);
	for (int kk = 0; kk < rel_num * 3; kk++){
		rel_vec_grad_sum[kk].resize(n, 0);
	}
	cout << "init entity adagrad vector and relation matrix adagrad is over." << endl;
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
	cout << "entity number = " << ent_num << endl;
	cout << "relation number = " << rel_num << endl;

	FILE *f_kb = fopen((model_base_path + "../data/train.txt").c_str(), "r");
	string sub, rel, obj;
	int sub_id, rel_id, obj_id;

	map<unsigned, set<unsigned> > rel_heads_tmp, rel_tails_tmp;
	map<unsigned, map<unsigned, set<unsigned> > > rel_head_tails;//计算平均一个head有多少个tail
	map<unsigned, map<unsigned, set<unsigned> > > rel_tail_heads;//计算平均一个tail有多少个head

	while (fscanf(f_kb, "%s", buf) == 1){
		sub = buf; fscanf(f_kb, "%s", buf); rel = buf; fscanf(f_kb, "%s", buf); obj = buf;		
		sub_id = ent2id[sub]; rel_id = rel2id[rel]; obj_id = ent2id[obj];
		train_tris_num++;
		train_tris.push_back(triple(sub_id, rel_id, obj_id));
		sub_rel_objs[sub_id][rel_id].insert(obj_id);
		rel_heads_tmp[rel_id].insert(sub_id);
		rel_tails_tmp[rel_id].insert(obj_id);
		rel_head_tails[rel_id][sub_id].insert(obj_id);
		rel_tail_heads[rel_id][obj_id].insert(sub_id);

	}
	fclose(f_kb);
	cout << "triple number = " << train_tris_num << endl;

	for (map<unsigned, set<unsigned> >::iterator iter = rel_heads_tmp.begin();
		iter != rel_heads_tmp.end(); iter++){
		unsigned rel_id = iter->first;
		for (set<unsigned>::iterator inner_iter = iter->second.begin(); inner_iter != iter->second.end(); inner_iter++)
			rel_heads[rel_id].push_back(*inner_iter);
	}
	for (map<unsigned, set<unsigned> >::iterator iter = rel_tails_tmp.begin();
		iter != rel_tails_tmp.end(); iter++){
		unsigned rel_id = iter->first;
		for (set<unsigned>::iterator inner_iter = iter->second.begin(); inner_iter != iter->second.end(); inner_iter++)
			rel_tails[rel_id].push_back(*inner_iter);
	}

	tail_num_per_head.resize(rel_num);
	head_num_per_tail.resize(rel_num);
	for (int rel_id = 0; rel_id < rel_num; rel_id++){
		//计算平均一个head有多少个tail
		map<unsigned, set<unsigned> > tails_per_head = rel_head_tails[rel_id];
		unsigned head_number = 0, tail_count = 0;
		for (map<unsigned, set<unsigned> > ::iterator iter = tails_per_head.begin();
			iter != tails_per_head.end(); iter++){
			if (iter->second.size() > 0){
				head_number++;
				tail_count += iter->second.size();
			}
		}
		tail_num_per_head[rel_id] = 1.0 * tail_count / head_number;
		//计算平均一个tail有多少个head
		map<unsigned, set<unsigned> > heads_per_tail = rel_tail_heads[rel_id];
		unsigned tail_number = 0, head_count = 0;
		for (map<unsigned, set<unsigned> > ::iterator iter = heads_per_tail.begin();
			iter != heads_per_tail.end(); iter++){
			if (iter->second.size() > 0){
				tail_number++;
				head_count += iter->second.size();
			}
		}
		head_num_per_tail[rel_id] = 1.0 * head_count / tail_number;
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
		subject = buf; fscanf(f_kb, "%s", buf);
		relation = buf; fscanf(f_kb, "%s", buf);
		object = buf;
		triple tri(ent2id[subject], rel2id[relation], ent2id[object]);
		triple_list.push_back(tri);
	}
	fclose(f_kb);
	cout << "load: " << filename << endl;
}
void eval_prepare(){
	if (Neg_Method == 1){
		load_eval_data(true, true, true, valid_pos);
		load_eval_data(true, false, true, valid_neg);
		load_eval_data(false, true, true, test_pos);
		load_eval_data(false, false, true, test_neg);
	}
	else{
		load_eval_data(true, true, false, valid_pos);
		load_eval_data(true, false, false, valid_neg);
		load_eval_data(false, true, false, test_pos);
		load_eval_data(false, false, false, test_neg);
	}	
}
double loss_triple(triple tri){
	unsigned sub = tri.h, rel = tri.r, obj = tri.t;
	vector<double> sub_vec = transHyper(ent_vec[sub], rel_vec[rel + 1]);
	vector<double> head_vec = add(sub_vec, rel_vec[rel]);
	vector<double> tail_vec = transHyper(ent_vec[obj], rel_vec[rel + 2]);
	if (L1_Flag == 1)
		return L1_distance(head_vec, tail_vec);
	else
		return L2_distance(head_vec, tail_vec);
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
	return eval_test();
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
	if ((i = ArgPos((char *)"-grad", argc, argv)) > 0) Grad_Method = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-rate", argc, argv)) > 0) rate = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-c", argc, argv)) > 0) c_value = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-margin", argc, argv)) > 0) margin = atof(argv[i + 1]);
	
	
	cout << "L1 / L2 = " << L1_Flag << endl;
	cout << "negative scope = " << Neg_Scope << endl;
	cout << "negative method = " << Neg_Method << endl;
	cout << "batch = " << Batch_Size << endl;
	cout << "epoch = " << Epoch_Size << endl;
	cout << "dim = " << n << endl;
	cout << "entity normalize weight(C) = " << c_value << endl;
	cout << "rate = " << rate << endl;
	cout << "margin = " << margin << endl;

	if (Neg_Method == 1)
		version = "bern";
	else
		version = "unif";
	char dim_ch[5];
	sprintf(dim_ch, "%d", n);
	dim_str = dim_ch;


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