#include "mcts.h"

MCTSTT::MCTSTT(size_t uct_hash_size) :_uct_hash_size(uct_hash_size), _used(0), _obsolete_game_ply(0)
{
	_uct_hash_mask = _uct_hash_size - 1;
	if (_uct_hash_mask & _uct_hash_size)
	{
		throw runtime_error("uct_hash_size must be power of 2");
	}
	entries = new NodeHashEntry[_uct_hash_size];
	nodes = new UCTNode[_uct_hash_size];
	clear();
}

void MCTSTT::clear()
{
	_used = 0;
	memset(entries, 0, sizeof(NodeHashEntry)*_uct_hash_size);
	memset(nodes, 0, sizeof(UCTNode)*_uct_hash_size);
}

UCTNode * MCTSTT::find_or_create_entry(const Position & pos, bool & created)
{
	return find_or_create_entry(pos.key(), pos.game_ply(), created);
}

UCTNode * MCTSTT::find_or_create_entry(Key key, int game_ply, bool & created)
{
	size_t orig_index = (size_t)key & _uct_hash_mask;
	size_t index = orig_index;
	while (true)
	{
		NodeHashEntry *nhe = &entries[index];
		if (nhe->flag)
		{
			if (nhe->game_ply < _obsolete_game_ply)
			{
				// ここを上書きする
				nhe->key = key;
				nhe->game_ply = game_ply;

				created = true;
				memset(&nodes[index], 0, sizeof(UCTNode));
				return &nodes[index];
			}
			if (nhe->key == key && nhe->game_ply == game_ply)
			{
				created = false;
				return &nodes[index];
			}

		}
		else
		{
			nhe->key = key;
			nhe->game_ply = game_ply;
			nhe->flag = true;
			_used++;
			created = true;
			return &nodes[index];
		}
		index = (index + 1) & _uct_hash_mask;
		if (index == orig_index)
		{
			// full
			return nullptr;
		}
	}
}

UCTNode * MCTSTT::find_entry(const Position & pos)
{
	return find_entry(pos.key(), pos.game_ply());
}

UCTNode * MCTSTT::find_entry(Key key, int game_ply)
{
	size_t orig_index = (size_t)key & _uct_hash_mask;
	size_t index = orig_index;
	while (true)
	{
		NodeHashEntry *nhe = &entries[index];
		if (nhe->flag)
		{
			if (nhe->key == key && nhe->game_ply == game_ply)
			{
				return &nodes[index];
			}
		}
		else
		{
			return nullptr;
		}
		index = (index + 1) & _uct_hash_mask;
		if (index == orig_index)
		{
			// full
			return nullptr;
		}
	}
	return nullptr;
}

int MCTSTT::get_hashfull() const
{
	return (int)(_used * 1000 / _uct_hash_size);
}

size_t MCTSTT::calc_uct_hash_size(int max_size_mb)
{
	size_t hash_size = (size_t)1 << MSB64(((unsigned long long)max_size_mb * 1024 * 1024) / (sizeof(NodeHashEntry) + sizeof(UCTNode)));
	return hash_size;
}

MCTSTT::~MCTSTT()
{
	delete[] entries;
	delete[] nodes;
}

MCTS::MCTS(size_t uct_hash_size) :c_puct(1.0), virtual_loss(1)
{
	tt = new MCTSTT(uct_hash_size);
}

MCTS::~MCTS()
{
	delete tt;
}

void MCTS::search(UCTNode * root, Position & pos, MCTSSearchInfo & sei, dnn_eval_obj *eval_info)
{
	sei.put_dnn_eval = false;
	sei.leaf_dup = false;
	sei.leaf_mate_search_found = false;
	mutex_.lock();
	sei.has_tt_lock = true;
	eval_info->index.path_length = 1;
	eval_info->index.path_indices[0] = root;
	search_recursive(root, pos, sei, eval_info);
	if (sei.has_tt_lock)
	{
		sei.has_tt_lock = false;
		mutex_.unlock();
	}
}


bool operator<(const dnn_move_index& left, const dnn_move_index& right) {
	// 確率で降順ソート用
	return left.prob > right.prob;
}

void MCTS::backup_dnn(dnn_eval_obj * eval_info)
{
	mutex_.lock();//ここはunique_lockを使ってもいい
	dnn_table_index &path = eval_info->index;
	// 末端ノードの評価を記録
	UCTNode &leaf_node = *path.path_indices[path.path_length - 1];
	leaf_node.evaled = true;
	// 事前確率でソートし、上位 MAX_UCT_CHILDREN だけ記録
	int n_moves_use = eval_info->n_moves;
	if (n_moves_use > MAX_UCT_CHILDREN)
	{
		std::sort(&eval_info->move_indices[0], &eval_info->move_indices[eval_info->n_moves]);
		n_moves_use = MAX_UCT_CHILDREN;
	}
	for (int i = 0; i < n_moves_use; i++)
	{
		dnn_move_index &dmi = eval_info->move_indices[i];
		leaf_node.move_list[i] = (Move)dmi.move;
		leaf_node.value_p[i] = dmi.prob;
		// n, w, qは0初期化されている
	}
	leaf_node.n_children = n_moves_use;
	float score = eval_info->static_value; // [-1.0, 1.0]
	if (eval_info->found_mate)
	{
		// この局面からの詰みが見つかっているため、DNNの評価に優先させる
		leaf_node.terminal = true;
		score = 1.0;//自分が攻め側
	}
	leaf_node.score = score;

	backup_tree(path, score);
	DupEvalChain *dec = leaf_node.dup_eval_chain;
	while (dec != nullptr)
	{
		backup_tree(dec->path, score);
		DupEvalChain *dec_next = dec->next;
		delete dec;
		dec = dec_next;
	}
	mutex_.unlock();
}

UCTNode * MCTS::make_root(Position & pos, MCTSSearchInfo & sei, dnn_eval_obj * eval_info, bool &created)
{
	mutex_.lock();
	UCTNode* root = tt->find_or_create_entry(pos, created);
	eval_info->index.path_indices[0] = root;
	eval_info->index.path_length = 1;
	sei.put_dnn_eval = false;

	if (created)
	{
		// 新規子ノードなので、評価
		float mate_score;
		bool not_mate = enqueue_pos(pos, sei, eval_info, mate_score);
		if (not_mate)
		{
			// 評価待ち
			// 非同期に処理される
			sei.put_dnn_eval = true;

		}
		else
		{
			// 詰んでいて評価対象にならない
			// ルートなのでbackupは不要
			// TODO
		}
	}
	else
	{
		// すでにあったのでそれを返す
	}
	mutex_.unlock();
	return root;
}

UCTNode * MCTS::get_root(const Position & pos)
{
	return tt->find_entry(pos);
}

Move MCTS::get_bestmove(UCTNode * root, Position & pos)
{
	Move bestMove = MOVE_RESIGN;
	float bestScore = -1;
	for (size_t i = 0; i < root->n_children; i++)
	{
		// 訪問回数で選択。それで決まらない場合は事前確率(0~1)で決める。
		float score = root->value_n[i] + root->value_p[i];
		if (score > bestScore)
		{
			bestScore = score;
			bestMove = root->move_list[i];
		}
	}
	return bestMove;
}

void MCTS::get_pv(UCTNode * root, Position & pos, std::vector<Move>& pv, float &winrate)
{
	std::lock_guard<std::mutex> lock(mutex_);
	get_pv_recursive(root, pos, pv, winrate, true);
}

int MCTS::get_hashfull()
{
	std::lock_guard<std::mutex> lock(mutex_);
	return tt->get_hashfull();
}

void MCTS::clear()
{
	tt->clear();
}

void MCTS::search_recursive(UCTNode * node, Position & pos, MCTSSearchInfo & sei, dnn_eval_obj *eval_info)
{
	if (eval_info->index.path_length >= MAX_SEARCH_PATH_LENGTH)
	{
		// 千日手模様の筋などで起こるかもしれないので一応対策
		// 引き分けとみなして終了する
		update_on_terminal(eval_info->index, 0.0);
		return;
	}

	if (node->terminal)
	{
		// 詰みノード
		// 評価は不要で、親へ評価値を再度伝播する
		update_on_terminal(eval_info->index, node->score);
		return;
	}

	if (eval_info->index.path_length > 1) // ルートノード自体を千日手とは判定しない
	{
		// 千日手判定。パスに依存するので、ノードには書き込まない。
		RepetitionState rep_state = pos.is_repetition(pos.game_ply() - eval_info->index.path_length);
		if (rep_state != RepetitionState::REPETITION_NONE)
		{
			float score;
			switch (rep_state)
			{
			case REPETITION_WIN:
			case REPETITION_SUPERIOR:
				score = 1.0;
				break;
			case REPETITION_LOSE:
			case REPETITION_INFERIOR:
				score = -1.0;
				break;
			default:
				score = 0.0;
				break;
			}

			update_on_terminal(eval_info->index, score);
			return;
		}
	}

	if (!node->evaled)
	{
		// ノードが評価中だった場合
		// virtual lossがあるので、評価が終わったときに追加でbackupを呼ぶようにする
		// link listにつなぐ
		DupEvalChain *dec = new DupEvalChain();
		memcpy(&dec->path, &eval_info->index, sizeof(dnn_table_index));
		dec->next = node->dup_eval_chain;
		node->dup_eval_chain = dec;
		sei.leaf_dup = true;
		return;
	}

	// エッジ選択
	size_t edge = select_edge(node);

	// virtual loss加算
	node->value_n[edge] += virtual_loss;
	node->value_n_sum += virtual_loss;
	node->value_w[edge] -= virtual_loss;

	Move m = node->move_list[edge];
	StateInfo si;
	pos.do_move(m, si);

	// 子ノードを選択するか生成
	bool created;
	UCTNode* child_node = tt->find_or_create_entry(pos, created);
	eval_info->index.path_child_indices[eval_info->index.path_length - 1] = (uint16_t)edge;
	eval_info->index.path_indices[eval_info->index.path_length] = child_node;
	eval_info->index.path_length++;

	if (created)
	{
		// 新規子ノードなので、評価
		float mate_score;
		// 行列作成前に置換表ロック開放
		sei.has_tt_lock = false;
		mutex_.unlock();
		bool not_mate = enqueue_pos(pos, sei, eval_info, mate_score);
		if (not_mate)
		{
			// 評価待ち
			// 非同期に処理される
			sei.put_dnn_eval = true;

			// 詰みがないか探索
			if (sei.mate_searcher)
			{
				std::vector<Move> moves;
				if (sei.mate_searcher->dfpn(pos, &moves))
				{
					// 詰みがある
					// DNNの結果の代わりに詰みであるという情報を入れることにする
					eval_info->found_mate = true;
					sei.leaf_mate_search_found = true;
				}
			}
		}
		else
		{
			// 詰んでいて評価対象にならない
			// 再度置換表をロックし直ちにbackup
			mutex_.lock();
			sei.has_tt_lock = true;
			update_on_mate(eval_info->index, mate_score);
		}
	}
	else
	{
		// 再帰的に探索
		search_recursive(child_node, pos, sei, eval_info);
	}

	pos.undo_move(m);
}

void MCTS::backup_tree(dnn_table_index & path, float leaf_score)
{
	float score = leaf_score;

	// treeをたどり値を更新
	for (int i = path.path_length - 2; i >= 0; i--)
	{
		//score = -score;
		score = score * -0.99F;//逃げる時はより長い詰み筋、追うときは短い詰み筋を選ぶよう調整
		UCTNode &inner_node = *path.path_indices[i];
		uint16_t edge = path.path_child_indices[i];
		int new_value_n = inner_node.value_n[edge] + 1 - virtual_loss;
		inner_node.value_n[edge] = new_value_n;
		float new_value_w = inner_node.value_w[edge] + score + virtual_loss;
		inner_node.value_w[edge] = new_value_w;
		// inner_node.vloss_ctr[edge]--;
		// inner_node.value_q[edge] = new_value_w / new_value_n;
		inner_node.value_n_sum += 1 - virtual_loss;
	}
}

void MCTS::update_on_terminal(dnn_table_index & path, float leaf_score)
{
	backup_tree(path, leaf_score);
}

void MCTS::update_on_mate(dnn_table_index & path, float mate_score)
{
	// 新規展開ノードがmateだったときの処理
	UCTNode &leaf_node = *path.path_indices[path.path_length - 1];
	leaf_node.evaled = true;
	leaf_node.terminal = true;
	leaf_node.score = mate_score;
	backup_tree(path, mate_score);

	// 末端ノードの詰み判定の最中に置換表ロックが外れるので、その間にdup_eval_chainにくっつけられる可能性がある
	DupEvalChain *dec = leaf_node.dup_eval_chain;
	while (dec != nullptr)
	{
		backup_tree(dec->path, mate_score);
		DupEvalChain *dec_next = dec->next;
		delete dec;
		dec = dec_next;
	}
}

size_t MCTS::select_edge(UCTNode * node)
{
	float n_sum_sqrt = sqrt((float)node->value_n_sum) + 0.001F;//完全に0だと最初の1手が事前確率に沿わなくなる
	size_t best_index = 0;
	float best_value = -100.0F;
	float w_sum = 0.0F;
	for (size_t i = 0; i < node->n_children; i++)
	{
		w_sum += node->value_w[i];
	}
	float mean_w = w_sum / node->value_n_sum;//1度も探索してないノードの評価値替わり
	for (size_t i = 0; i < node->n_children; i++)
	{
		int value_n = node->value_n[i];
		float value_u = node->value_p[i] / (value_n + 1) * c_puct * n_sum_sqrt;
		float value_q = value_n > 0 ? node->value_w[i] / value_n : mean_w;
		float value_sum = value_q + value_u;
		if (value_sum > best_value)
		{
			best_value = value_sum;
			best_index = i;
		}
	}

	return best_index;
}

bool MCTS::enqueue_pos(const Position & pos, MCTSSearchInfo & sei, dnn_eval_obj *eval_info, float & score)
{
	if (pos.DeclarationWin() != MOVE_NONE)
	{
		// 勝ち局面
		score = 1.0;
		return false;
	}
	//if (use_mate_search && mate_search_leaf->dfpn(pos, nullptr))
	//{
	//	// 詰めて勝てる局面
	//	score = 1.0;
	//	mate_search_leaf_count++;
	//	return false;
	//}

	// 局面を評価用の行列にする。その際詰みであることが判明した場合、DNN評価しない。

	int m_i = 0;
	bool not_mate = false;
	for (auto m : MoveList<LEGAL>(pos))
	{
		dnn_move_index &dmi = eval_info->move_indices[m_i];
		dmi.move = (uint16_t)m.move;
		dmi.index = (uint16_t)sei.cvt->get_move_index(pos, m.move);
		m_i++;
		not_mate = true;
	}

	if (not_mate)
	{
		eval_info->found_mate = false;
		eval_info->n_moves = m_i;
		sei.cvt->get_board_array(pos, eval_info->input_array);
		eval_info->response_queue = sei.response_queue;
		sei.request_queue->push(eval_info);
		score = 0.0; //dummy
		return true;
	}
	else
	{
		// 詰みなので、DNN評価はせず直ちにbackupする。
		score = -1.0;
		return false;
	}
}

void MCTS::get_pv_recursive(UCTNode * node, Position & pos, std::vector<Move>& pv, float & winrate, bool root)
{
	if (node->terminal)
	{
		if (root)
		{
			winrate = node->score;
		}
		return;
	}
	int best_n = -1;
	Move bestMove = MOVE_RESIGN;
	int best_child_i = 0;
	for (int i = 0; i < node->n_children; i++)
	{
		if (node->value_n[i] > best_n)
		{
			best_n = node->value_n[i];
			bestMove = node->move_list[i];
			best_child_i = i;
		}
	}
	if (pos.pseudo_legal(bestMove) && pos.legal(bestMove))
	{
		pv.push_back(bestMove);
		StateInfo si;
		pos.do_move(bestMove, si);
		UCTNode* child_node = tt->find_entry(pos);
		if (child_node)
		{
			get_pv_recursive(child_node, pos, pv, winrate, false);
		}
		else
		{
			// 読み筋が途切れた
			// 詰まないものとして扱う
		}
		pos.undo_move(bestMove);
	}
	if (root)
	{
		winrate = node->value_w[best_child_i] / node->value_n[best_child_i];
	}
}

void UCTNode::pprint()
{
	sync_cout << "info string node ";
	if (terminal)
	{
		cout << "terminal ";
	}
	if (!evaled)
	{
		cout << "not_evaled ";
	}
	cout << "visited " << value_n_sum << " ";
	cout << "score " << score << " ";
	for (int i = 0; i < n_children; i++)
	{
		cout << move_list[i] << " " << value_n[i] << "," << value_w[i] << "," << value_p[i] << " ";
	}
	cout << sync_endl;
}
