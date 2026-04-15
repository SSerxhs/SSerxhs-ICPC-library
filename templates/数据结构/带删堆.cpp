template<class T, class T1 = vector<T>, class T2 = less<T>> struct heap
{
private:
	priority_queue<T, T1, T2> p, q;
public:
	void push(const T &x)
	{
		if (!q.empty() && q.top() == x)
		{
			q.pop();
			while (!q.empty() && q.top() == p.top()) p.pop(), q.pop();
		}
		else p.push(x);
	}
	void pop()
	{
		p.pop();
		while (!q.empty() && p.top() == q.top()) p.pop(), q.pop();
	}
	void pop(const T &x)
	{
		if (p.top() == x)
		{
			p.pop();
			while (!q.empty() && p.top() == q.top()) p.pop(), q.pop();
		}
		else q.push(x);
	}
	T top() const { return p.top(); }
	int size() const { return p.size() - q.size(); }
	bool empty() const { return p.empty(); }
	vector<T> to_vector() const
	{
		vector<T> a;
		auto P = p, Q = q;
		while (P.size())
		{
			a.push_back(P.top()); P.pop();
			while (Q.size() && P.top() == Q.top()) P.pop(), Q.pop();
		}
		return a;
	}
};


