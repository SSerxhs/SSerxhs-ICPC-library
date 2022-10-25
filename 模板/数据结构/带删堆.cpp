template<typename T> struct heap
{
	priority_queue<T> p,q;
	void push(const T &x)
	{
		if (!q.empty()&&q.top()==x)
		{
			q.pop();
			while (!q.empty()&&q.top()==p.top()) p.pop(),q.pop();
		} else p.push(x);
	}
	void pop()
	{
		p.pop();
		while (!q.empty()&&p.top()==q.top()) p.pop(),q.pop(); 
	}
	void pop(const T &x)
	{
		if (p.top()==x)
		{
			p.pop();
			while (!q.empty()&&p.top()==q.top()) p.pop(),q.pop(); 
		} else q.push(x);
	}
	T top() {return p.top();}
	bool empty() {return p.empty();}
};
