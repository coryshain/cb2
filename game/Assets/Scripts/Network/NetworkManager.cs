using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Web;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
using System.Linq;


namespace Network
{
    public class NetworkManager : MonoBehaviour
    {
        public static string TAG = "NetworkManager";

        public static NetworkManager Instance;

        public readonly static string URL = "ws://localhost:8080/player_endpoint";

        private ClientConnection _client;
        private NetworkMapSource _networkMapSource;
        private NetworkRouter _router;
        private EntityManager _entityManager;
        private Player _player;
        private DateTime _lastStatsPoll;
        private Role _role = Network.Role.NONE;
        private Role _currentTurn = Network.Role.NONE;

        public IMapSource MapSource()
        {
            Scene scene = SceneManager.GetActiveScene();
            Debug.Log("[DEBUG] scene: " + scene.name);
            if (scene.name == "menu_scene")
            {
                Debug.Log("Loading menu map");
                return new FixedMapSource();
            }
            if (_networkMapSource == null)
            {
                Debug.Log("Retrieved map source before it was initialized.");
            }
            return _networkMapSource;
        }

        public static NetworkManager TaggedInstance()
        {
            GameObject obj = GameObject.FindGameObjectWithTag(Network.NetworkManager.TAG);
            if (obj == null)
                return null;
            return obj.GetComponent<Network.NetworkManager>();
        }

        public static string BaseUrl(bool webSocket=true)
        {
            string url = URL;
            if (Application.absoluteURL != "")
            {
                // We can figure out the server's address based on Unity's API.
                Uri servedUrl = new Uri(Application.absoluteURL);

                string scheme = servedUrl.Scheme;
                if (webSocket) {
                    scheme = servedUrl.Scheme == "https" ? "wss" : "ws";
                }
                UriBuilder endpointUrlBuilder =
                    new UriBuilder(scheme, servedUrl.Host, servedUrl.Port);
                if (servedUrl.Query.Length > 0)
                {
                    endpointUrlBuilder.Query = servedUrl.Query.Substring(1);  // Remove leading '?'
                }
                url = endpointUrlBuilder.Uri.AbsoluteUri;
            }
            return url;
        }

        public static Dictionary<string, string> UrlParameters()
        {
            // We can figure out the server's address based on Unity's API.
            Uri servedUrl = new Uri(Application.absoluteURL);
            string query = servedUrl.Query;
            // Remove the initial ?.
            if (query.Length > 0 && query[0] == '?')
            {
                query = query.Substring(1);
            }
            NameValueCollection urlParameters = HttpUtility.ParseQueryString(query);
            // Convert the NameValueCollection to a Dictionary<string, string>.
            return urlParameters.AllKeys.ToDictionary(t => t, t => urlParameters[t]);
        }


        public Role Role()
        {
            return _role;
        }

        public Role CurrentTurn()
        {
            return _currentTurn;
        }

        public void TransmitAction(ActionQueue.IAction action)
        {
            _router.TransmitAction(action);
        }

        public void RespondToPing()
        {
            MessageToServer toServer = new MessageToServer();
            toServer.transmit_time = DateTime.Now.ToString("o");
            toServer.type = MessageToServer.MessageType.PONG;
            toServer.pong = new Pong{ping_receive_time = DateTime.Now.ToString("o")};
            _client.TransmitMessage(toServer);
        }

        public void TransmitObjective(ObjectiveMessage objective)
        {
            MessageToServer toServer = new MessageToServer();
            toServer.transmit_time = DateTime.Now.ToString("o");
            toServer.type = MessageToServer.MessageType.OBJECTIVE;
            toServer.objective = objective;
            toServer.objective.sender = _role;
            _client.TransmitMessage(toServer);
        }

        public void TransmitLiveFeedback(LiveFeedback feedback)
        {
            MessageToServer toServer = new MessageToServer();
            toServer.transmit_time = DateTime.Now.ToString("o");
            toServer.type = MessageToServer.MessageType.LIVE_FEEDBACK;
            toServer.live_feedback = feedback;
            _client.TransmitMessage(toServer);
        }

        public void TransmitObjectiveComplete(ObjectiveCompleteMessage objectiveComplete)
        {
            MessageToServer toServer = new MessageToServer();
            toServer.transmit_time = DateTime.Now.ToString("o");
            toServer.type = MessageToServer.MessageType.OBJECTIVE_COMPLETE;
            toServer.objective_complete = objectiveComplete;
            _client.TransmitMessage(toServer);
        }

        public void TransmitTurnComplete()
        {
            MessageToServer toServer = new MessageToServer();
            toServer.transmit_time = DateTime.Now.ToString("o");
            toServer.type = MessageToServer.MessageType.TURN_COMPLETE;
            _client.TransmitMessage(toServer);
        }

        public void Awake()
        {
            gameObject.tag = TAG;
        }

        // Called when a user clicks the "Join Game" menu button. Enters the game queue.
        public void JoinGame()
        {
            MessageToServer msg = new MessageToServer();
            msg.transmit_time = DateTime.Now.ToString("o");
            msg.type = MessageToServer.MessageType.ROOM_MANAGEMENT;
            msg.room_request = new RoomManagementRequest();
            msg.room_request.type = RoomRequestType.JOIN;
            Debug.Log("[DEBUG]Joining game...");
            _client.TransmitMessage(msg);
        }

        public void StartLeaderTutorial()
        {
            StartTutorial(TutorialRequest.LEADER_TUTORIAL);
        }

        public void StartFollowerTutorial()
        {
            StartTutorial(TutorialRequest.FOLLOWER_TUTORIAL);
        }

        public void StartTutorial(string tutorialName)
        {
            MessageToServer msg = new MessageToServer();
            msg.transmit_time = DateTime.Now.ToString("o");
            msg.type = MessageToServer.MessageType.TUTORIAL_REQUEST;
            msg.tutorial_request = new TutorialRequest();
            msg.tutorial_request.type = TutorialRequestType.START_TUTORIAL;
            msg.tutorial_request.tutorial_name = tutorialName;
            Debug.Log("[DEBUG]Joining tutorial...");
            _client.TransmitMessage(msg);
        }

        public void NextTutorialStep()
        {
            MessageToServer msg = new MessageToServer();
            msg.transmit_time = DateTime.Now.ToString("o");
            msg.type = MessageToServer.MessageType.TUTORIAL_REQUEST;
            msg.tutorial_request = new TutorialRequest();
            msg.tutorial_request.type = Network.TutorialRequestType.REQUEST_NEXT_STEP;
            Debug.Log("[DEBUG]Requesting next tutorial step...");
            _client.TransmitMessage(msg);            
        }

        // Pulls the player out of the wait queue to join a new game.
        public void CancelGameQueue()
        {
            MessageToServer msg = new MessageToServer();
            msg.transmit_time = DateTime.Now.ToString("o");
            msg.type = MessageToServer.MessageType.ROOM_MANAGEMENT;
            msg.room_request = new RoomManagementRequest();
            msg.room_request.type = RoomRequestType.CANCEL;
            _client.TransmitMessage(msg);
        }

        // Leave an active game -- signals to the server that we are leaving.
        public void QuitGame()
        {
            MessageToServer msg = new MessageToServer();
            msg.transmit_time = DateTime.Now.ToString("o");
            msg.type = MessageToServer.MessageType.ROOM_MANAGEMENT;
            msg.room_request = new RoomManagementRequest();
            msg.room_request.type = RoomRequestType.LEAVE;
            _client.TransmitMessage(msg);
            ReturnToMenu();
        }

        // Return to the main menu.
        public void ReturnToMenu()
        {
            _networkMapSource.ClearMapUpdate();
            _role = Network.Role.NONE;
            _currentTurn = Network.Role.NONE;
            _router.SetEntityManager(null);
            _router.SetPlayer(null);
            SceneManager.LoadScene("menu_scene");
        }

        // Display the Game Over screen, with an optional explanation.
        public void DisplayGameOverMenu(string reason="")
        {
            MenuTransitionHandler.TaggedInstance().DisplayEndGameMenu(reason);
        }

        public Util.Status InitializeTaggedObjects()
        {
            GameObject obj = GameObject.FindGameObjectWithTag(EntityManager.TAG);
            if (obj == null)
            {
                _entityManager = null;
                return Util.Status.NotFound("Could not find tag: " + EntityManager.TAG);
            }
            _entityManager = obj.GetComponent<EntityManager>();
            if (_entityManager == null)
            {
                return Util.Status.NotFound("Could not find component: " + EntityManager.TAG);
            }

            GameObject playerObj = GameObject.FindGameObjectWithTag(Player.TAG);
            if (playerObj == null)
            {
                _player = null;
                return Util.Status.NotFound("Could not find tag: " + Player.TAG);
            }
            _player = playerObj.GetComponent<Player>();
            if (_player == null)
            {
                return Util.Status.NotFound("Could not find component: " + Player.TAG);
            }

            _router.SetEntityManager(_entityManager);
            _router.SetPlayer(_player);
            return Util.Status.OkStatus();
        }

        // Start is called before the first frame update
        private void Start()
        {
            if (Instance == null)
            {
                Instance = this;
                DontDestroyOnLoad(this);  // Persist network connection between scene changes.
            }
            else if (Instance != this)
            {
                Destroy(gameObject);
                return;
            }
            _networkMapSource = new NetworkMapSource();

            string url = URL;
            if (Application.absoluteURL != "")
            {
                // We can figure out the server's address based on Unity's API.
                Uri servedUrl = new Uri(Application.absoluteURL);
                string websocketScheme = servedUrl.Scheme == "https" ? "wss" : "ws";
                UriBuilder endpointUrlBuilder =
                    new UriBuilder(websocketScheme, servedUrl.Host, servedUrl.Port,
                                   "/player_endpoint");
                if (servedUrl.Query.Length > 0)
                {
                    endpointUrlBuilder.Query = servedUrl.Query.Substring(1);  // Remove leading '?'
                }
                url = endpointUrlBuilder.Uri.AbsoluteUri;
            }
            Debug.Log("Using url: " + url);
            _client = new ClientConnection(url, /*autoReconnect=*/ true);
            _router = new NetworkRouter(_client, _networkMapSource, this, null, null);

            Util.Status result = InitializeTaggedObjects();
            if (!result.Ok())
            {
                Debug.Log(result);
            }

            // Subscribe to new scene changes.
            SceneManager.sceneLoaded += OnSceneLoaded;


            _lastStatsPoll = DateTime.Now;
            _client.Start();
        }

        public void OnSceneLoaded(Scene scene, LoadSceneMode mode)
        {
            if (scene.name == "menu_scene")
                return;    
            Util.Status result = InitializeTaggedObjects();
            if (!result.Ok())
            {
                Debug.Log(result);
            }
        }

        public void OnApplicationQuit()
        {
            _client.OnApplicationQuit();
        }

        public void HandleTutorialResponse(TutorialResponse tutorialResponse)
        {
            if (tutorialResponse.type == TutorialResponseType.STARTED)
            {
                Debug.Log("[DEBUG]Tutorial started.");
                SceneManager.LoadScene("tutorial_scene");
                _role = tutorialResponse.Role();
            }
            else if (tutorialResponse.type == TutorialResponseType.COMPLETED)
            {
                Debug.Log("[DEBUG]Tutorial completed.");
                DisplayGameOverMenu("Tutorial completed. Your participation has been recorded.");
            }
            else if (tutorialResponse.type == TutorialResponseType.STEP)
            {
                Debug.Log("[DEBUG]Tutorial next step received.");
                TutorialManager.TaggedInstance().HandleTutorialStep(tutorialResponse.step);
            }
        }

        public void HandleRoomManagement(RoomManagementResponse response)
        {
            if (response.type == RoomResponseType.JOIN_RESPONSE)
            {
                if (response.join_response.joined)
                {
                    Debug.Log("Joined room as " + response.join_response.role + "!");
                    SceneManager.LoadScene("game_scene");
                    _role = response.join_response.role;
                }
                else
                {
                    Debug.Log("Waiting for room. Position in queue: " + response.join_response.place_in_queue);
                }
            }
            else if (response.type == RoomResponseType.LEAVE_NOTICE)
            {
                Debug.Log("Game ended. Reason: " + response.leave_notice.reason);
                Scene scene = SceneManager.GetActiveScene();
                Debug.Log("[DEBUG] scene: " + scene.name);
                if (scene.name != "menu_scene")
                {
                    DisplayGameOverMenu("Game ended. Reason: " + response.leave_notice.reason);
                }
            }
            else if (response.type == RoomResponseType.STATS)
            {
                Debug.Log("Stats: " + response.stats.ToString());
                GameObject obj = GameObject.FindGameObjectWithTag("Stats");
                if (obj == null) return;
                Text stats = obj.GetComponent<Text>();
                stats.text = "Players in game: " + response.stats.players_in_game + "\n" +
                             "Games: " + response.stats.number_of_games + "\n" +
                             "Players Waiting: " + response.stats.players_waiting + "\n";
            }
            else if (response.type == RoomResponseType.ERROR)
            {
                Debug.Log("Received room management error: " + response.error);
            }
            else
            {
                Debug.Log("Received unknown room management response type: " + response.type);
            }
        }

        public void HandleTurnState(TurnState state)
        {
            _currentTurn = state.turn;
        }

        // Update is called once per frame
        void Update()
        {
            GameObject statsObj = GameObject.FindGameObjectWithTag("Stats");
            if (((DateTime.Now - _lastStatsPoll).Seconds > 1) && (statsObj != null) && (statsObj.activeInHierarchy))
            {
                Debug.Log("Requesting stats..");
                _lastStatsPoll = DateTime.Now;
                MessageToServer msg = new MessageToServer();
                msg.transmit_time = DateTime.Now.ToString("o");
                msg.type = MessageToServer.MessageType.ROOM_MANAGEMENT;
                msg.room_request = new RoomManagementRequest();
                msg.room_request.type = RoomRequestType.STATS;
                _client.TransmitMessage(msg);
            }

            Text connectionStatus = GameObject.FindGameObjectWithTag("ConnectionStatus").GetComponent<Text>();
            if (_client.IsClosed())
            {
                connectionStatus.text = "Disconnected";
            }
            else if (_client.IsConnected())
            {
                connectionStatus.text = "";
            }
            else if (_client.IsConnecting())
            {
                connectionStatus.text = "Connecting...";
            }
            else if (_client.IsClosing())
            {
                connectionStatus.text = "Closing...";
            }

            _client.Update();
        }
    }
}
