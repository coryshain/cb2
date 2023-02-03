using Network;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TMPro;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using Newtonsoft.Json;

public class MenuTransitionHandler : MonoBehaviour
{
    public static readonly string TAG = "InGameMenuHandler";

    public enum MenuState
    {
        NONE,
        ESCAPE_MENU,
    }

    private static readonly string ESCAPE_MENU_TAG = "ESCAPE_MENU";
    private static readonly string INPUT_FIELD_TAG = "MessageInputField";
    private static readonly string OBJECTIVE_LIST = "OBJECTIVE_LIST";
    private static readonly string SCROLL_VIEW_TAG = "ScrollView";

    private static readonly string GAME_OVER_MENU = "GAME_OVER_UI";
    private static readonly string GAME_OVER_STATS = "GAME_OVER_STATS";
    private static readonly string GAME_OVER_REASON = "GAME_OVER_REASON";

    private static readonly string SCORE_TEXT_TAG = "SCORE_TEXT";
    private static readonly string LEADER_SCORE_TEXT_TAG = "LEADER_SCORE_TEXT";
    private static readonly string FOLLOWER_SCORE_TEXT_TAG = "FOLLOWER_SCORE_TEXT";
    private static readonly string OUR_TURN_TAG = "OUR_TURN_INDICATOR";
    private static readonly string NOT_OUR_TURN_TAG = "NOT_OUR_TURN_INDICATOR";

    private static readonly string END_TURN_PANEL = "END_TURN_PANEL";

    private static readonly string POSITIVE_FEEDBACK_TAG = "THUMBS_UP_SIGNAL";
    private static readonly string NEGATIVE_FEEDBACK_TAG = "THUMBS_DOWN_SIGNAL";

    private static readonly string FEEDBACK_WINDOW_TAG = "FEEDBACK_WINDOW";

    private static readonly string TURN_START_SOUND = "TURN_START_SOUND";

    private static readonly string MUTE_AUDIO_TOGGLE = "MUTE_AUDIO_TOGGLE";

    // Used for replay scene.
    private static readonly string FOLLOWER_TURN_TAG = "FOLLOWER_TURN_INDICATOR";
    private static readonly string LEADER_TURN_TAG = "LEADER_TURN_INDICATOR";

    private static readonly string LEADER_FEEDBACK_WINDOW = "LEADER_FEEDBACK_WINDOW";

    private static readonly string LOGIN_STATUS_PANEL = "LOGIN_STATUS_PANEL";
    private static readonly string LOGIN_STATUS_TEXT = "LOGIN_STATUS_TEXT";

    private static readonly string DYNAMIC_MENU_BUTTONS = "DYNAMIC_BUTTONS";

    private static readonly string INFO_BULLETIN = "BULLETIN_TEXT";

    public static MenuTransitionHandler Instance;
    private Logger _logger;

    // We re-use ActionQueue here to animate UI transparency. It's a bit
    // overkill to have two animation queues here, but it's very obvious what's
    // happening for the reader, and that's worth it.
    private ActionQueue notOurTurnIndicatorFade = new ActionQueue();
    private ActionQueue ourTurnIndicatorFade = new ActionQueue();
    // Used for replays.
    private ActionQueue leaderTurnIndicatorFade = new ActionQueue();
    private ActionQueue followerTurnIndicatorFade = new ActionQueue();

    private MenuState _currentMenuState;

    private DateTime _lastTurnTransmitTime;
    private DateTime _lastTurnRefreshTime;
    private TurnState _lastTurn = null;

    List<Network.ObjectiveMessage> _lastObjectivesList = new List<Network.ObjectiveMessage>();

    private DateTime _lastPositiveFeedback = DateTime.MinValue;
    private GameObject _positiveFeedbackSignal;
    private DateTime _lastNegativeFeedback = DateTime.MinValue;
    private GameObject _negativeFeedbackSignal;

    private static readonly float FEEDBACK_DURATION_SECONDS = 1.0f;

    public static MenuTransitionHandler TaggedInstance()
    {
        if (Instance != null) {
            return Instance;
        }
        GameObject obj = GameObject.FindGameObjectWithTag(MenuTransitionHandler.TAG);
        if (obj == null)
            return null;
        return obj.GetComponent<MenuTransitionHandler>();
    }

    public void QuitGame()
    {
        Network.NetworkManager.TaggedInstance().QuitGame();
    }

    [DllImport("__Internal")]
    private static extern void DownloadJson(string filename, string data);

    public void SaveGameData()
    {
        // Downloads the game's map update to a json file.
        Network.NetworkManager networkManager = Network.NetworkManager.TaggedInstance();
        IMapSource mapSource = networkManager.MapSource();
        if (mapSource == null)
        {
            Debug.Log("No map source.");
            return;
        }
        Network.MapUpdate mapUpdate = mapSource.RawMapUpdate();

        List<TurnState> turnStateLog = new List<TurnState>();
        turnStateLog.Add(_lastTurn);

        Network.BugReport localBugReport = new Network.BugReport();
        localBugReport.MapUpdate = mapUpdate;
        localBugReport.TurnStateLog = turnStateLog;

        localBugReport.Logs = new List<ModuleLog>();
        List<string> modules = Logger.GetTrackedModules();
        Debug.Log("Modules: " + modules.Count);
        foreach (string module in modules)
        {
            Logger logger = Logger.GetTrackedLogger(module);
            ModuleLog moduleLog = new ModuleLog();
            moduleLog.Module = module;
            moduleLog.Log = System.Text.Encoding.UTF8.GetString(logger.GetBuffer());
            localBugReport.Logs.Add(moduleLog);
            Debug.Log("Module: " + module + " and log size: " + moduleLog.Log.Length);
        }

        string bugReportJson = JsonConvert.SerializeObject(localBugReport, Formatting.Indented);
        DownloadJson("client_bug_report.json.log", bugReportJson);
    }

    public delegate void UploadCallback(string contents);

    private UploadCallback _uploadCallback;

    [DllImport("__Internal")]
    private static extern void PromptUpload();

    public void AskForAFile(UploadCallback callback)
    {
        _uploadCallback = callback;
        PromptUpload();
    }

    public void OnFileReady(string contents)
    {
        if (_uploadCallback == null)
        {
            Debug.LogError("No callback set for file upload.");
            return;
        }
        _uploadCallback(contents);
    }

    public void SaveScenarioData(string scenario_data)
    {
        DownloadJson("scenario_state.json", scenario_data);
    }

    public void BackToMenu()
    {
        Network.NetworkManager.TaggedInstance().ReturnToMenu();
    }

    public List<Network.ObjectiveMessage> ObjectiveList()
    {
        return _lastObjectivesList;
    }


    public void RenderObjectiveList(List<Network.ObjectiveMessage> objectives)
    {
        _lastObjectivesList = objectives;
        GameObject objectivesPanel = GameObject.FindGameObjectWithTag(MenuTransitionHandler.OBJECTIVE_LIST);
        if (objectivesPanel == null)
        {
            Debug.LogError("Could not find objective list panel");
            return;
        }
        
        foreach (Transform child in objectivesPanel.transform)
        {
            Destroy(child.gameObject);
        }

        Network.NetworkManager networkManager = Network.NetworkManager.TaggedInstance();

        int activeIndex = -1;
        for(int i = 0; i < objectives.Count; ++i)
        {
            UnityAssetSource source = new UnityAssetSource();
            GameObject uiTemplate;
            if (objectives[i].completed)
            {
                uiTemplate = source.LoadUi(IAssetSource.UiId.OBJECTIVE_COMPLETE);
            } else if (objectives[i].cancelled) {
                uiTemplate = source.LoadUi(IAssetSource.UiId.OBJECTIVE_CANCELLED);
            } else if (activeIndex == -1) {
                activeIndex = i;
                uiTemplate = source.LoadUi(IAssetSource.UiId.OBJECTIVE_ACTIVE);
            } else {
                uiTemplate = source.LoadUi(IAssetSource.UiId.OBJECTIVE_PENDING);
            }
            GameObject objectiveUi = Instantiate(uiTemplate) as GameObject;
            objectiveUi.transform.SetParent(objectivesPanel.transform);
            objectiveUi.transform.localScale = Vector3.one;
            objectiveUi.transform.localPosition = Vector3.zero;
            objectiveUi.transform.localRotation = Quaternion.identity;
            objectiveUi.transform.Find("Label").gameObject.GetComponent<TMPro.TMP_Text>().text = objectives[i].text;
            if (activeIndex == i)
            {
                objectiveUi.GetComponent<UIObjectiveInfo>().Objective = objectives[i];
            }
            if ((activeIndex != -1) && (activeIndex < i))
            {
                if (networkManager.Role() == Network.Role.LEADER)
                {
                    objectiveUi.transform.Find("Label").gameObject.GetComponent<TMPro.TMP_Text>().text = "(unseen) " + objectives[i].text;
                } else {
                    objectiveUi.transform.Find("Label").gameObject.GetComponent<TMPro.TMP_Text>().text = "(pending objective)";
                    // Only draw one "(pending objective)", even if multiple are available.
                    break;
                }
            }
        }

        Canvas.ForceUpdateCanvases();
        GameObject scrollObj = GameObject.FindGameObjectWithTag(MenuTransitionHandler.SCROLL_VIEW_TAG);
        if (scrollObj == null)
        {
            Debug.LogError("Could not find scroll view");
            return;
        }

        ScrollRect sRect = scrollObj.GetComponent<ScrollRect>();
        StartCoroutine(ApplyScrollPosition(sRect, 0.0f));
    }

    private IEnumerator ApplyScrollPosition(ScrollRect sr, float verticalPos)
    {
        yield return new WaitForEndOfFrame();
        sr.verticalNormalizedPosition = verticalPos;
        LayoutRebuilder.ForceRebuildLayoutImmediate((RectTransform)sr.transform);
    }

    public void OnCompleteObjective(ObjectiveCompleteMessage complete)
    {
        Network.NetworkManager networkManager = Network.NetworkManager.TaggedInstance();
        networkManager.TransmitObjectiveComplete(complete);
    }

    public void CancelPendingObjectives()
    {
        Network.NetworkManager networkManager = Network.NetworkManager.TaggedInstance();
        networkManager.TransmitCancelPendingObjectives();
    }

    public void SendObjective()
    {
        Network.ObjectiveMessage objective = new Network.ObjectiveMessage();

        // Get the text entered by the user.
        GameObject textObj = GameObject.FindWithTag(INPUT_FIELD_TAG);
        TMPro.TMP_InputField textMeshPro = textObj.GetComponent<TMPro.TMP_InputField>();
        if (textMeshPro.text.Length == 0)
        {
            Debug.Log("No objective text entered.");
            return;
        }

        Network.NetworkManager networkManager = Network.NetworkManager.TaggedInstance();

        objective.text = textMeshPro.text;
        objective.sender = networkManager.Role();
        objective.completed = false;

        networkManager.TransmitObjective(objective);

        // Clear the text field and unselect it.
        textMeshPro.text = "";
        EventSystem.current.SetSelectedGameObject(null);
    }

    public void SendPositiveFeedback()
    {
        if (!Network.NetworkManager.TaggedInstance().ServerConfig().live_feedback_enabled)
        {
            _logger.Info("SendPositiveFeedback(): Live feedback is not enabled.");
            return;
        }
        Network.LiveFeedback feedback = new Network.LiveFeedback();
        feedback.signal = Network.FeedbackType.POSITIVE;
        Network.NetworkManager.TaggedInstance().TransmitLiveFeedback(feedback);
    }

    public void SendNegativeFeedback()
    {
        if (!Network.NetworkManager.TaggedInstance().ServerConfig().live_feedback_enabled)
        {
            _logger.Info("SendNegativeFeedback(): Live feedback is not enabled.");
            return;
        }
        Network.LiveFeedback feedback = new Network.LiveFeedback();
        feedback.signal = Network.FeedbackType.NEGATIVE;
        Network.NetworkManager.TaggedInstance().TransmitLiveFeedback(feedback);
    }

    public void HandleLiveFeedback(LiveFeedback feedback)
    {
        _logger.Info("Received feedback: " + feedback.signal);
        // Display the positive feedback signal for 2 seconds.
        if (feedback.signal == Network.FeedbackType.POSITIVE)
        {
            _lastPositiveFeedback = DateTime.Now;
        } else if (feedback.signal == Network.FeedbackType.NEGATIVE)
        {
            _lastNegativeFeedback = DateTime.Now;
        }
    }

    public void HandleTurnState(DateTime transmitTime, Network.TurnState state)
    {
        _logger.Info("Received turn state: " + state);
        if (state.game_over)
        {
            EndGame(transmitTime, state);
        }
        else
        {
            DisplayTurnState(transmitTime, state);
        }
    }

    public static void LoginStarted()
    {
        Logger logger = Logger.GetOrCreateTrackedLogger(TAG);
        Scene scene = SceneManager.GetActiveScene();
        if (scene.name != "menu_scene") {
            logger.Info("LoginStarted(): Not in menu scene, ignoring.");
            return;
        }
        GameObject login_status_panel = GameObject.FindWithTag(LOGIN_STATUS_PANEL);
        if (login_status_panel == null)
        {
            logger.Error("Could not find login status panel.");
            return;
        }
        login_status_panel.SetActive(true);
        TMP_Text login_status = FindTmpTextWithTag(LOGIN_STATUS_TEXT);
        if (login_status == null)
        {
            logger.Error("Could not find login status text.");
            return;
        }
        login_status.text = "Please wait... Logging In.";
    }

    // Only valid if called while in menu scene. This is a bit unkosher as this
    // game object doesn't even exist in the menu scene, so we make it a static
    // helper function so it can be called whenever (and this method doesn't
    // make use of any Unity GameObjects)
    public static void HandleLoginStatus(GoogleAuthConfirmation auth)
    {
        Logger logger = Logger.GetOrCreateTrackedLogger(TAG);
        logger.Info("Received login status: " + auth);
        Scene scene = SceneManager.GetActiveScene();
        if (scene.name != "menu_scene") {
            logger.Info("HandleLoginStatus(): Not in menu scene, ignoring.");
            return;
        }
        GameObject login_status_panel = GameObject.FindWithTag(LOGIN_STATUS_PANEL);
        if (login_status_panel == null)
        {
            logger.Error("Could not find login status panel.");
            return;
        }
        login_status_panel.SetActive(true);
        if (auth.success)
        {
            TMP_Text login_status = FindTmpTextWithTag(LOGIN_STATUS_TEXT);
            login_status.text = "Login successful. Hit Play to start!";
            logger.Info("Login successful.");
        }
        else
        {
            TMP_Text login_status = FindTmpTextWithTag(LOGIN_STATUS_TEXT);
            login_status.text = "Login failed. Please try again another time.";
            logger.Info("Login failed.");
        }
    }

    private void DisplayTurnStateReplayMode(DateTime transmitTime, Network.TurnState state)
    {
        NetworkManager networkManager = Network.NetworkManager.TaggedInstance();
        string twoLineSummary = state.ShortStatus(networkManager.Role(), networkManager.IsReplay());
        if (state.turn == Network.Role.LEADER)
        {
            GameObject scoreObj = GameObject.FindWithTag(LEADER_SCORE_TEXT_TAG);
            TMPro.TMP_Text textMeshPro = scoreObj.GetComponent<TMPro.TMP_Text>();
            textMeshPro.text = twoLineSummary;
        } else {
            GameObject scoreObj = GameObject.FindWithTag(FOLLOWER_SCORE_TEXT_TAG);
            TMPro.TMP_Text textMeshPro = scoreObj.GetComponent<TMPro.TMP_Text>();
            textMeshPro.text = twoLineSummary;
        }

        if ((_lastTurn != null) && (_lastTurn.turn != state.turn))
        {
            Debug.Log("Changing turn animation. " + _lastTurn.turn + " -> " + state.turn);
            GameObject endTurnPanel = GameObject.FindGameObjectWithTag(END_TURN_PANEL);
            // If we're in fast forward mode, we don't want to queue up a bunch
            // of turn transitions. Flush the queue so only the most recent one
            // shows.
            followerTurnIndicatorFade.Flush();
            leaderTurnIndicatorFade.Flush();
            if (state.turn == Network.Role.LEADER)
            {
                followerTurnIndicatorFade.AddAction(Fade.FadeOut(0.5f));
                leaderTurnIndicatorFade.AddAction(Instant.Pause(0.5f));
                leaderTurnIndicatorFade.AddAction(Fade.FadeIn(0.5f));
            } else {
                Debug.Log("TURN follower fade in.");
                leaderTurnIndicatorFade.AddAction(Fade.FadeOut(0.5f));
                followerTurnIndicatorFade.AddAction(Instant.Pause(0.5f));
                followerTurnIndicatorFade.AddAction(Fade.FadeIn(0.5f));
            }
        }
        _lastTurnTransmitTime = transmitTime;
        _lastTurn = state;

        // Force the canvas to re-render in order to display the new text mesh.
        Canvas.ForceUpdateCanvases();
    }

    private void UpdateTurnUIText(Network.TurnState state)
    {
        Network.NetworkManager networkManager = Network.NetworkManager.TaggedInstance();
        string twoLineSummary = state.ShortStatus(networkManager.Role(), networkManager.IsReplay());
        GameObject scoreObj = GameObject.FindWithTag(SCORE_TEXT_TAG);
        if (scoreObj == null)
        {
            _logger.Warn("Could not find score text object.");
            return;
        }
        TMPro.TMP_Text textMeshPro = scoreObj.GetComponent<TMPro.TMP_Text>();
        textMeshPro.text = twoLineSummary;
    }

    private void DisplayTurnState(DateTime transmitTime, Network.TurnState state)
    {
        Network.NetworkManager networkManager = Network.NetworkManager.TaggedInstance();

        if (networkManager.IsReplay())
        {
            DisplayTurnStateReplayMode(transmitTime, state);
            return;
        }

        UpdateTurnUIText(state);

        if ((_lastTurn != null) && (_lastTurn.turn != state.turn))
        {
            Debug.Log("Changing turn animation. " + _lastTurn.turn + " -> " + state.turn);
            GameObject endTurnPanel = GameObject.FindGameObjectWithTag(END_TURN_PANEL);
            if (state.turn == networkManager.Role())
            {
                notOurTurnIndicatorFade.AddAction(Fade.FadeOut(0.5f));
                ourTurnIndicatorFade.AddAction(Instant.Pause(0.5f));
                ourTurnIndicatorFade.AddAction(Fade.FadeIn(0.5f));
                if (endTurnPanel != null)
                {
                    endTurnPanel.transform.localScale = new Vector3(1f, 1f, 1f);
                }
            } else {
                ourTurnIndicatorFade.AddAction(Fade.FadeOut(0.5f));
                notOurTurnIndicatorFade.AddAction(Instant.Pause(0.5f));
                notOurTurnIndicatorFade.AddAction(Fade.FadeIn(0.5f));
                if (endTurnPanel != null)
                {
                    endTurnPanel.transform.localScale = new Vector3(0f, 0f, 0f);
                }
            }
            if (_lastTurn.turn != Role.NONE && state.turn == networkManager.Role())
            {
                GameObject soundObject = GameObject.FindGameObjectWithTag(TURN_START_SOUND);
                if (soundObject != null)
                {
                    AudioSource audioSource = soundObject.GetComponent<AudioSource>();
                    if (audioSource != null)
                    {
                        bool playAudio = true;
                        GameObject muteToggleObject = GameObject.FindGameObjectWithTag(MUTE_AUDIO_TOGGLE);
                        if (muteToggleObject != null)
                        {
                            Toggle muteToggle = muteToggleObject.GetComponent<Toggle>();
                            if (muteToggle != null)
                            {
                                Debug.Log("Mute toggle is " + muteToggle.isOn);
                                playAudio = !muteToggle.isOn;
                            }
                        }
                        if (playAudio)
                        {
                            audioSource.Play();
                        }
                    }
                }
            }
        }
        _lastTurnTransmitTime = transmitTime;
        _lastTurn = state;

        // Force the canvas to re-render in order to display the new text mesh.
        Canvas.ForceUpdateCanvases();
    }

    public static Canvas FindCanvasWithTag(string tag)
    {
        GameObject obj = GameObject.FindGameObjectWithTag(tag);
        if (obj == null)
        {
            Debug.Log("Unable to find canvas with tag: " + tag);
            return null;
        }
        return obj.GetComponent<Canvas>();
    }

    public static Text FindTextWithTag(string tag)
    {
        GameObject obj = GameObject.FindGameObjectWithTag(tag);
        if (obj == null)
        {
            Debug.Log("Unable to find text with tag: " + tag);
            return null;
        }
        return obj.GetComponent<Text>();
    }

    public static TMPro.TMP_Text FindTmpTextWithTag(string tag)
    {
        GameObject obj = GameObject.FindGameObjectWithTag(tag);
        if (obj == null)
        {
            Debug.Log("Unable to find text with tag: " + tag);
            return null;
        }
        return obj.GetComponent<TMPro.TMP_Text>();
    }

    Button FindButtonWithTag(string tag)
    {
        GameObject obj = GameObject.FindGameObjectWithTag(tag);
        if (obj == null)
        {
            Debug.Log("Unable to find button with tag: " + tag);
            return null;
        }
        return obj.GetComponent<Button>();
    }

    // Displays the end game menu. Optionally display an explanation.
    public void DisplayEndGameMenu(string reason="")
    {
        Canvas gameOverCanvas = FindCanvasWithTag(GAME_OVER_MENU);
        if (gameOverCanvas == null)
        {
            Debug.Log("Unable to find end game menu.");
            return;
        }
        if (gameOverCanvas.enabled)
        {
            // Don't do anything if the end game menu is already displayed. Just log the reason.
            Debug.Log("Attempted to display the end game menu while already enabled. Reason: " + reason);
            return;
        }

        Text reasonText = FindTextWithTag(GAME_OVER_REASON);
        reasonText.text = reason;
        EndGame(_lastTurnTransmitTime, _lastTurn);
    }

    [DllImport("__Internal")]
    private static extern void SubmitMturk(string game_data);

    private void EndGame(DateTime transmitTime, Network.TurnState state)
    {
        // Hide escape menu.
        GameObject esc_menu = GameObject.FindWithTag(ESCAPE_MENU_TAG);
        _currentMenuState = MenuState.NONE;
        esc_menu.GetComponent<Canvas>().enabled = false;
        
        Canvas gameOverCanvas = FindCanvasWithTag(GAME_OVER_MENU);
        gameOverCanvas.enabled = true;

        Text score = FindTextWithTag(GAME_OVER_STATS);
        score.text = state.ScoreString(transmitTime);
    }

    void Awake()
    {
        _logger = Logger.GetOrCreateTrackedLogger("MenuTransitionHandler");
    }

    // Start is called before the first frame update
    void Start()
    {
        if (Instance == null)
        {
            Instance = this;
            DontDestroyOnLoad(this);  // Persist network connection between scene changes.
        } else if (Instance != this) {
            _logger.Warn("Tried to create duplicate menu object. Self-destructed game object.");
            DestroyImmediate(gameObject);
            return;
        }
        // Subscribe to new scene changes.
        SceneManager.sceneLoaded += OnSceneLoaded;
        _currentMenuState = MenuState.NONE;
        notOurTurnIndicatorFade = new ActionQueue("NotOurTurnQueue");
        ourTurnIndicatorFade = new ActionQueue("OurTurnQueue");
        _lastTurnRefreshTime = DateTime.MinValue;
    }

    public void OnSceneLoaded(Scene scene, LoadSceneMode mode)
    {
        if (scene.name != "menu_scene") {
            _positiveFeedbackSignal = GameObject.FindWithTag(POSITIVE_FEEDBACK_TAG);
            _negativeFeedbackSignal = GameObject.FindWithTag(NEGATIVE_FEEDBACK_TAG);
        }
    }

    public void TurnComplete()
    {
        Network.NetworkManager.TaggedInstance().TransmitTurnComplete();
    }

    public void DisplayMenu(MenuOptions m)
    {
        _logger.Info("Displaying menu.");
        Scene scene = SceneManager.GetActiveScene();
        if (scene.name != "menu_scene") {
            _logger.Warn("Attempted to set menu in non-menu scene.");
            return;
        }

        _logger.Info("Bulletin");
        // Set the information bulletin.
        Text bulletin = FindTextWithTag(INFO_BULLETIN);
        bulletin.text = m.bulletin_message;

        // Remove dynamic menu children.
        _logger.Info("Clearing dynamic menu.");
        GameObject dynamic_menu = GameObject.FindWithTag(DYNAMIC_MENU_BUTTONS);
        foreach (Transform child in dynamic_menu.transform) {
            Destroy(child.gameObject);
        }

        // Add new menu buttons from prefab 
        _logger.Info("Adding new dynamic buttons.");
        UnityAssetSource assetSource = new UnityAssetSource();
        GameObject prefab = assetSource.LoadUi(IAssetSource.UiId.MENU_BUTTON);
        if (prefab == null)
        {
            _logger.Warn("Unable to load menu button prefab.");
            return;
        }
        foreach (Network.ButtonDescriptor button in m.menu_buttons)
        {
            GameObject ui_button = Instantiate(prefab);
            ui_button.transform.SetParent(dynamic_menu.transform, false);
            ui_button.GetComponent<Text>().text = button.text;
            ui_button.GetComponent<Button>().onClick.AddListener(() => {
                ButtonUtils.HandleAction(button.code); 
            });
        }
    }

    // Update is called once per frame
    void Update()
    {
        Scene scene = SceneManager.GetActiveScene();
        if (scene.name == "menu_scene") {
            // Do nothing in menu scene.
            return;
        }
        NetworkManager networkManager = Network.NetworkManager.TaggedInstance();
        if (networkManager.IsReplay())
        {
            // Handle replay UI animations.
            leaderTurnIndicatorFade.Update();
            State.Continuous lTS = leaderTurnIndicatorFade.ContinuousState();
            GameObject leader_turn_obj = GameObject.FindWithTag(LEADER_TURN_TAG);
            CanvasGroup leader_turn_group = leader_turn_obj.GetComponent<CanvasGroup>();
            leader_turn_group.alpha = lTS.Opacity;

            followerTurnIndicatorFade.Update();
            State.Continuous fTS = followerTurnIndicatorFade.ContinuousState();
            GameObject follower_turn_obj = GameObject.FindWithTag(FOLLOWER_TURN_TAG);
            CanvasGroup follower_turn_group = follower_turn_obj.GetComponent<CanvasGroup>();
            follower_turn_group.alpha = fTS.Opacity;
        } else {
            // Handle UI animations.
            ourTurnIndicatorFade.Update();
            State.Continuous tS = ourTurnIndicatorFade.ContinuousState();
            GameObject turn_obj = GameObject.FindWithTag(OUR_TURN_TAG);
            CanvasGroup turn_group = turn_obj.GetComponent<CanvasGroup>();
            turn_group.alpha = tS.Opacity;

            notOurTurnIndicatorFade.Update();
            State.Continuous nTS = notOurTurnIndicatorFade.ContinuousState();
            GameObject not_turn_obj = GameObject.FindWithTag(NOT_OUR_TURN_TAG);
            CanvasGroup not_turn_group = not_turn_obj.GetComponent<CanvasGroup>();
            not_turn_group.alpha = nTS.Opacity;
        }
        // Refresh the menu UI showing the most recent turn state (so that time remaining stays fresh). Once per second.
        if (DateTime.Now - _lastTurnRefreshTime > TimeSpan.FromSeconds(1))
        {
            _lastTurnRefreshTime = DateTime.Now;
            if (_lastTurn != null) {
                UpdateTurnUIText(_lastTurn);
            }
        }

        if (Input.GetKeyDown(KeyCode.Return))
        {
            SendObjective();
        }

        if (Input.GetKeyDown(KeyCode.T) && !UserTypingInput())
        {
            GameObject textObj = GameObject.FindWithTag(INPUT_FIELD_TAG);
            TMPro.TMP_InputField textMeshPro = textObj.GetComponent<TMPro.TMP_InputField>();
            textMeshPro.Select();
        }

        if (Input.GetKeyDown(KeyCode.N) && !UserTypingInput())
        {
            TurnComplete();
        }

        GameObject esc_menu = GameObject.FindWithTag(ESCAPE_MENU_TAG);
        if (esc_menu == null)
        {
            Debug.Log("Could not find escape menu!");
            return;
        }

        if (Input.GetKeyDown(KeyCode.Escape))
        {
            if (_currentMenuState == MenuState.NONE)
            {
                Canvas gameOverCanvas = FindCanvasWithTag(GAME_OVER_MENU);
                if (gameOverCanvas.enabled)
                {
                    // Don't do anything if the end game menu is already displayed.
                    return;
                }
                _currentMenuState = MenuState.ESCAPE_MENU;
                esc_menu.GetComponent<Canvas>().enabled = true;
                Debug.Log("Opening esc menu");
            }
            else if (_currentMenuState == MenuState.ESCAPE_MENU)
            {
                _currentMenuState = MenuState.NONE;
                esc_menu.GetComponent<Canvas>().enabled = false;
                Debug.Log("Closed esc menu");
            }
        }

        // Handle feedback UI.
        UnityEngine.Color feedbackColor = new UnityEngine.Color(1f, 1f, 1f, 100f / 255f);  // Neutral color.
        if (_positiveFeedbackSignal == null && networkManager.Role() == Role.FOLLOWER)
            _positiveFeedbackSignal = GameObject.FindWithTag(POSITIVE_FEEDBACK_TAG);
        if (_negativeFeedbackSignal == null && networkManager.Role() == Role.FOLLOWER)
            _negativeFeedbackSignal = GameObject.FindWithTag(NEGATIVE_FEEDBACK_TAG);
        bool is_pos = (DateTime.Now - _lastPositiveFeedback).TotalSeconds < FEEDBACK_DURATION_SECONDS;
        bool is_neg = (DateTime.Now - _lastNegativeFeedback).TotalSeconds < FEEDBACK_DURATION_SECONDS;

        // If only one is active, show it.
        if (!(is_pos && is_neg)) {
            if (_positiveFeedbackSignal)
                _positiveFeedbackSignal.SetActive(is_pos);
            if (_negativeFeedbackSignal)
                _negativeFeedbackSignal.SetActive(is_neg);
            if (is_pos)
                feedbackColor = new UnityEngine.Color(0f, 1f, 0f, 1f);  // Green.
            if (is_neg)
                feedbackColor = new UnityEngine.Color(1f, 0f, 0f, 1f);  // Red.
        }

        // If both are active, only show the most recent.
        if (is_pos && is_neg) {
            if (_lastPositiveFeedback < _lastNegativeFeedback) 
            {
                feedbackColor = new UnityEngine.Color(1f, 0f, 0f, 1f);  // Red.
                if (_negativeFeedbackSignal)
                    _negativeFeedbackSignal.SetActive(true);
                if (_positiveFeedbackSignal)
                    _positiveFeedbackSignal.SetActive(false);
            } else {
                if (_positiveFeedbackSignal)
                    _positiveFeedbackSignal.SetActive(true);
                if (_negativeFeedbackSignal)
                    _negativeFeedbackSignal.SetActive(false);
                feedbackColor = new UnityEngine.Color(0f, 1f, 0f, 1f);  // Green.
            }
        }

        GameObject leader_feedback_window = GameObject.FindWithTag(LEADER_FEEDBACK_WINDOW);
        if (leader_feedback_window != null)
            leader_feedback_window.GetComponent<Image>().color = feedbackColor;
        GameObject feedback_obj = GameObject.FindWithTag(FEEDBACK_WINDOW_TAG);
        if (feedback_obj != null)
            feedback_obj.GetComponent<Image>().color = feedbackColor;
    }

    private bool UserTypingInput()
    {
        return EventSystem.current.currentSelectedGameObject != null;
    }
}
