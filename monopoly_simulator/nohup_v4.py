loss_train ===> tensor(0.0039, device='cuda:0', grad_fn=<DivBackward0>)
Remove monopoly_simple-v1 from registry
Step # :0, avg score : 0.133
Step # :0, avg winning : 0.000
loss_train ===> tensor(0.0594, device='cuda:0', grad_fn=<DivBackward0>)
Step # :10000, avg score : 0.133
Step # :10000, avg winning : 0.000
loss_train ===> tensor(0.0494, device='cuda:0', grad_fn=<DivBackward0>)
Step # :20000, avg score : 0.133
Step # :20000, avg winning : 0.000
loss_train ===> tensor(0.0497, device='cuda:0', grad_fn=<DivBackward0>)
Step # :30000, avg score : 0.133
Step # :30000, avg winning : 0.000
loss_train ===> tensor(0.0477, device='cuda:0', grad_fn=<DivBackward0>)
Step # :40000, avg score : 0.133
Step # :40000, avg winning : 0.000
loss_train ===> tensor(0.0341, device='cuda:0', grad_fn=<DivBackward0>)
Step # :50000, avg score : 0.133
Step # :50000, avg winning : 0.000
loss_train ===> tensor(0.0243, device='cuda:0', grad_fn=<DivBackward0>)
Step # :60000, avg score : 0.133
Step # :60000, avg winning : 0.000
loss_train ===> tensor(0.0138, device='cuda:0', grad_fn=<DivBackward0>)
Step # :70000, avg score : 0.133
Step # :70000, avg winning : 0.000
Traceback (most recent call last):
  File "vanilla_A2C_main_v4.py", line 270, in <module>
    # a = np.zeros([2,2])
  File "vanilla_A2C_main_v4.py", line 146, in train
    values.append(self.model.critic(s))
  File "/media/becky/GNOME-p3/monopoly_simulator/vanilla_A2C.py", line 133, in step_nochange
    return self.step_wait()
  File "/media/becky/GNOME-p3/monopoly_simulator/vanilla_A2C.py", line 117, in step_wait
    results = [master_end.recv() for master_end in self.master_ends] #receive from worker_end #format???
  File "/media/becky/GNOME-p3/monopoly_simulator/vanilla_A2C.py", line 117, in <listcomp>
    results = [master_end.recv() for master_end in self.master_ends] #receive from worker_end #format???
  File "/home/eilab/anaconda3/lib/python3.7/multiprocessing/connection.py", line 250, in recv
    buf = self._recv_bytes()
  File "/home/eilab/anaconda3/lib/python3.7/multiprocessing/connection.py", line 407, in _recv_bytes
    buf = self._recv(4)
  File "/home/eilab/anaconda3/lib/python3.7/multiprocessing/connection.py", line 379, in _recv
    chunk = read(handle, remaining)
KeyboardInterrupt
