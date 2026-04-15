using ll = long long;
template<class T, class T1 = vector<T>, class T2 = less<T>> struct ksum_pop
{
private:
	struct __cmp
	{
		bool operator()(const T &x, const T &y) const
		{
			return x != y && !T2()(x, y);
		}
	};
	heap<T, T1, __cmp> p;
	heap<T, T1, T2> q;
	ll cur;
public:
	ksum_pop() :cur(0) { }
	void push(const T &x)
	{
		if (!q.size() || !T2()(x, q.top())) p.push(x), cur += x; else q.push(x);
	}
	int size() const { return p.size() + q.size(); }
	void pop(const T &x)
	{
		if (q.size() && !T2()(q.top(), x)) q.pop(x);
		else p.pop(x), cur -= x;
	}
	ll sum(int k)
	{
		while (p.size() < k)
		{
			cur += q.top();
			p.push(q.top());
			q.pop();
		}
		while (p.size() > k)
		{
			cur -= p.top();
			q.push(p.top());
			p.pop();
		}
		return cur;
	}
};

