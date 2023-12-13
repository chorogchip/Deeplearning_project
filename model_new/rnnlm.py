# coding: utf-8
import sys
sys.path.append('..')
from common.time_layers import *
from common.base_model import BaseModel


class Rnnlm_new(BaseModel):
    def __init__(self, vocab_size=196006, wordvec_size=100, hidden_size=100):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # 가중치 초기화
        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx1 = (rn(D, 10 * H) / np.sqrt(D)).astype('f')
        lstm_Wx2 = (rn(D, H) / np.sqrt(D)).astype('f')
        lstm_Wx3 = (rn(D, H) / np.sqrt(D)).astype('f')
        lstm_Wh1 = (rn(H, 10 * H) / np.sqrt(H)).astype('f')
        lstm_Wh2 = (rn(H, H) / np.sqrt(H)).astype('f')
        lstm_Wh3 = (rn(H, H) / np.sqrt(H)).astype('f')
        lstm_b1= np.zeros(10 * H).astype('f')
        lstm_b2 = np.zeros(H).astype('f')
        lstm_b3 = np.zeros(H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        # 계층 생성
        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM_new(lstm_Wx1, lstm_Wx2, lstm_Wx3,
                     lstm_Wh1, lstm_Wh2, lstm_Wh3,
                     lstm_b1, lstm_b2, lstm_b3,
                     stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer = self.layers[1]

        # 모든 가중치와 기울기를 리스트에 모은다.
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts):
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.lstm_layer.reset_state()
