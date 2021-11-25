using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using NativeWebSocket;
using Newtonsoft.Json;
using UnityEngine;

namespace Network
{

    public class ClientConnection
    {
        private WebSocket _webSocket;
        public string _url;
        private NetworkRouter _router;
        private ConcurrentQueue<Network.MessageToServer> _messageQueue;

        string fix_json(string value)
        {
            value = "{\"Items\":" + value + "}";
            return value;
        }

        public ClientConnection(string url)
        {
            _url = url;
            _webSocket = new WebSocket(_url);
            _messageQueue = new ConcurrentQueue<Network.MessageToServer>();
        }

        public void RegisterHandler(NetworkRouter router)
        {
            _router = router;
        }

        public bool IsClosed()
        {
            return _webSocket.State.HasFlag(WebSocketState.Closed);
        }

        public bool IsConnected()
        {
            return _webSocket.State.HasFlag(WebSocketState.Open);
        }
        public bool IsConnecting()
        {
            return _webSocket.State.HasFlag(WebSocketState.Connecting);
        }

        public bool IsClosing()
        {
            return _webSocket.State.HasFlag(WebSocketState.Closing);
        }

        public void TransmitMessage(MessageToServer message)
        {
            _messageQueue.Enqueue(message);
        }

        public async void Reconnect()
        {
            _webSocket.OnOpen += () =>
            {
                Debug.Log("Connection open!");
            };

            _webSocket.OnError += (e) =>
            {
                Debug.Log("Error! " + e);
            };

            _webSocket.OnClose += (e) =>
            {
                Debug.Log("Connection closed! Reconnecting...");
            };

            _webSocket.OnMessage += (bytes) =>
            {
                if (_router == null)
                {
                    return;
                }

                string received = System.Text.Encoding.ASCII.GetString(bytes);

                MessageFromServer message = JsonConvert.DeserializeObject<MessageFromServer>(System.Text.Encoding.ASCII.GetString(bytes));
                _router.HandleMessage(message);
            };

            // waiting for messages
            await _webSocket.Connect();
        }

        // Start is called before the first frame update
        public void Start()
        {
            Reconnect();
        }

        public void Update()
        {
            SendPendingActions();
#if !UNITY_WEBGL || UNITY_EDITOR
            _webSocket.DispatchMessageQueue();
#endif
        }

        private async void SendPendingActions()
        {
            if (_webSocket.State == WebSocketState.Open)
            {
                if (!_messageQueue.TryDequeue(out MessageToServer toServer))
                {
                    return;
                }

                if (toServer == null)
                {
                    Debug.Log("Dequeued a null MessageToServer.");
                    return;
                }

                Debug.Log("Sending: " + JsonUtility.ToJson(toServer));
                await _webSocket.SendText(JsonUtility.ToJson(toServer));
            }
        }

        public async void OnApplicationQuit()
        {
            await _webSocket.Close();
        }

    }

}  // namespace Network