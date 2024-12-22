import socketio
from abc import ABC, abstractmethod
from typing import List, Optional
from logicai.entities import TeamsPromptInput, TeamsPromptOutput

__all__ = [
    'entities',
    'logging',
    'tools',
    'entities',
    'LogicAIBase',
    'SocketIOLogicAI'
]




class LogicAIBase(ABC):


    @abstractmethod
    def on_request(self, prompt: TeamsPromptInput) -> None:
        """AI'a istek gönder.

        Args:
            prompt (TeamsPromptInput): İstek metni ve gönderilecek dosyaların yolunu belirtir.
        """
        pass

    @abstractmethod
    def emit_response(self, output: TeamsPromptOutput) -> None:
        """_summary_

        Args:
            output (TeamsPromptOutput): _description_
        """
        pass

    @abstractmethod
    def on_like(self, message: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def on_dislike(self, message: Optional[str] = None) -> None:
        pass

    @abstractmethod
    def on_clear(self) -> None:
        pass

class AgentBase(ABC):
    """Bu sınıftan nesne oluşan nesneyle geliştirici kendi chat agent'ını tanımlayacak.

    Args:
        ABC (_type_): _description_
    """

    _agents = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        AgentBase._agents.append(cls)

    @abstractmethod
    def request(self, prompt: TeamsPromptInput) -> TeamsPromptOutput:
        """Agent'a istek gönder.

        Args:
            prompt (TeamsPromptInput): Prompt metni ve yüklenen dosyaların path'(ler)i

        Returns:
            TeamsPromptOutput: Dönen cevap prompt'u ve indirilecek dosyaların path'i
        """
        pass

    @abstractmethod
    def like(self, message: Optional[str] = None) -> None:
        """Son konuşmayı onaylar. Yapay zeka eğitimleri için log kaydeder.

        Args:
            message (Optional[str], optional): Like mesajı. Defaults to None.
        """
        pass

    @abstractmethod
    def dislike(self, message: Optional[str] = None) -> None:
        """Son konuşmayı onaylamaz. Yapay zeka eğitimleri için log kaydeder.

        Args:
            message (Optional[str], optional): Dislike mesajı. Defaults to None.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Yapay zekanın hafızasını temizler.
        """
        pass



class SocketIOLogicAI(LogicAIBase):
    def __init__(self, sio: socketio.Client):
        self.sio = sio
        self._register_socketio_events()

    def _register_socketio_events(self) -> None:
        """Tüm `on_` ile başlayan metodları Socket.IO olaylarıyla bağlar."""
        # Tüm sınıf metodlarını kontrol et
        for attr_name in dir(self):
            if attr_name.startswith("on_"):
                method = getattr(self, attr_name)
                if callable(method):
                    # Socket.IO olay adını `on_` sonrası kısmıyla bağla
                    event_name = attr_name[3:]
                    self.sio.on(event_name, method)

    def on_request(self, prompt: TeamsPromptInput) -> None:
        print(f"Received request: {prompt}")

    def emit_response(self, output: TeamsPromptOutput) -> None:
        self.sio.emit("response", output)

    def on_like(self, message: Optional[str] = None) -> None:
        print(f"Liked: {message}")

    def on_dislike(self, message: Optional[str] = None) -> None:
        print(f"Disliked: {message}")

    def on_clear(self) -> None:
        print("Cleared")
